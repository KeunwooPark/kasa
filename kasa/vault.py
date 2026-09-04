"""The vault — the one place on this machine a secret is allowed to be.

`config.toml` still holds no secrets: it names the environment variable that
carries one, and `docs/DESIGN.md` §11.2 stays true. But an env var only covers
the credentials Kasa is *started* with, and an agent working with a user
acquires authorization mid-conversation — a token pasted into a DM, a key for a
service it was just given access to. Those have nowhere to go today except the
transcript, which is the one place they must never be.

So the vault is not really storage. It is a containment boundary. A value
inside it is:

- known to `kasa.redact.Redactor`, so it is matched exactly rather than by
  shape at every outbound boundary — logs, tool results, recalled memory;
- refused by the memory patch validator, so it cannot reach a commit;
- never rendered into a model prompt.

Everything else in Kasa can then be made to refuse secrets, because there is
somewhere for them to go instead.

**It never leaves the machine.** No commit, no push, no database row, no sync.
Kasa must never ask anyone to put this file in git, and that is enforced here
rather than documented: the vault refuses to load from inside the memory repo,
and `kasa doctor` warns when it is inside any git work tree at all.

Threat model, stated plainly so nobody mistakes what this buys. The file is
plaintext at `0600` in a `0700` directory. That defends against *committed to
git*, *synced to a dotfiles repo*, and *read by another user on the box*. It
does not defend against root, or against someone who already has the ability to
run code as this user — and neither would the alternatives: a passphrase defeats
a daemon that has to restart unattended, and a key file stored next to the
ciphertext is theatre. Better to be honest about the line than to draw it in a
place that only looks further away.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from kasa.errors import ConfigError

#: Overrides where the vault is looked for, as `KASA_CONFIG` and `KASA_DB` do.
VAULT_ENV = "KASA_VAULT"

VERSION = 1

UNSAFE_PERMISSIONS = (
    "{path} is readable by other users (mode {mode}).\n"
    "It holds credentials, so Kasa will not read it until that is fixed:\n"
    "    chmod 600 {path}"
)
UNSAFE_DIRECTORY = (
    "{path} is accessible by other users (mode {mode}).\n"
    "The vault directory must be private; fix it with:\n"
    "    chmod 700 {path}"
)

INSIDE_MEMORY_REPO = (
    "{path} is inside the long-term memory repo at {clone}.\n"
    "That repo is committed and pushed to GitHub, and the vault never leaves "
    "this machine. Move it — {default} is the default — or set {env}."
)

NOT_AN_OBJECT = "{path} is not a Kasa vault: expected a JSON object, found {found}."


def vault_path() -> Path:
    """Where the vault lives.

    The *data* directory, deliberately, and not `~/.config/kasa/` beside
    `config.toml`: people sync `~/.config` into dotfiles repositories, which
    would walk the vault straight into the kind of push this exists to prevent.
    `config.toml` is safe to commit and this is not, so they do not live
    together.
    """
    if override := os.environ.get(VAULT_ENV):
        return Path(override).expanduser()
    return Path(user_data_dir("kasa")) / "vault.json"


def fingerprint(value: str) -> str:
    """A short digest, so two machines can be compared without revealing anything.

    Truncated to 12 hex characters: enough to tell "these are the same key"
    from "these are different keys", and far too little to attack the value
    through. The full digest would be a hash of a high-entropy secret, which is
    not a disclosure either, but there is no reason to print more than answers
    the question.
    """
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Entry:
    """One secret, without its value. What `kasa vault list` is allowed to see."""

    name: str
    fingerprint: str
    updated: str


class Vault:
    """The secrets on this machine, keyed by the env var name that would carry them.

    Keying by env var name rather than inventing a second namespace is what
    makes this backward compatible: an existing `key_env = "ANTHROPIC_API_KEY"`
    keeps working untouched, and `kasa vault set ANTHROPIC_API_KEY` is simply
    another way of supplying it.
    """

    def __init__(self, path: Path, secrets: dict[str, dict[str, str]] | None = None) -> None:
        self.path = path
        self._secrets = secrets or {}

    # -- reading -------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | None = None) -> Vault:
        """Read the vault. A missing file is an empty vault, not an error.

        Unsafe permissions *are* an error. Reading a credential out of a
        world-readable file and carrying on would make the mode meaningless,
        and this is the only moment Kasa is in a position to notice.
        """
        target = path or vault_path()
        try:
            stat = target.stat()
        except FileNotFoundError:
            return cls(target)
        except OSError as exc:
            raise ConfigError(f"{target} could not be read: {exc}") from exc

        directory_mode = target.parent.stat().st_mode & 0o777
        if directory_mode & 0o077:
            raise ConfigError(
                UNSAFE_DIRECTORY.format(path=target.parent, mode=format(directory_mode, "03o"))
            )
        mode = stat.st_mode & 0o777
        if mode & 0o077:
            raise ConfigError(UNSAFE_PERMISSIONS.format(path=target, mode=format(mode, "03o")))

        try:
            raw = target.read_text()
        except OSError as exc:
            raise ConfigError(f"{target} could not be read: {exc}") from exc

        return cls(target, _parse(raw, target))

    def get(self, name: str) -> str | None:
        entry = self._secrets.get(name)
        return entry.get("value") if entry else None

    def __contains__(self, name: str) -> bool:
        return name in self._secrets

    def names(self) -> list[str]:
        return sorted(self._secrets)

    def entries(self) -> list[Entry]:
        """Every secret, without its value."""
        return [
            Entry(
                name=name,
                fingerprint=fingerprint(entry.get("value", "")),
                updated=entry.get("updated", "unknown"),
            )
            for name, entry in sorted(self._secrets.items())
        ]

    def values(self) -> dict[str, str]:
        """Name to value, for seeding the redactor. The only caller that needs these."""
        return {name: entry["value"] for name, entry in self._secrets.items() if entry.get("value")}

    # -- writing -------------------------------------------------------------

    def set(self, name: str, value: str) -> None:
        if not name:
            raise ConfigError("a secret needs a name")
        if not value:
            raise ConfigError(f"refusing to store an empty value for {name}")
        self._secrets[name] = {
            "value": value,
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def remove(self, name: str) -> bool:
        return self._secrets.pop(name, None) is not None

    def save(self) -> None:
        """Write the vault, never passing through a moment of being readable.

        Opened with an explicit `0o600` rather than written and then chmod'd:
        between the write and the chmod, a file created under the default umask
        is a credential on disk that another user can read, and that window is
        exactly what an attacker with a loop is waiting for. The rename is
        atomic, so a crash mid-write leaves the previous vault intact rather
        than a truncated one.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        payload = json.dumps({"version": VERSION, "secrets": self._secrets}, indent=2) + "\n"

        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        # `os.replace` carries the temp file's mode across, but an existing
        # vault with looser permissions would have kept its own; setting it
        # explicitly means saving a vault also repairs one.
        os.chmod(self.path, 0o600)


def _parse(raw: str, path: Path) -> dict[str, dict[str, str]]:
    try:
        loaded: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigError(NOT_AN_OBJECT.format(path=path, found=type(loaded).__name__))

    secrets = loaded.get("secrets", {})
    if not isinstance(secrets, dict):
        raise ConfigError(f"{path} is not a Kasa vault: 'secrets' is not an object.")

    parsed: dict[str, dict[str, str]] = {}
    for name, entry in secrets.items():
        # A bare string is accepted as the value so a vault can be written by
        # hand in a pinch; it is normalized on the next save.
        if isinstance(entry, str):
            parsed[str(name)] = {"value": entry, "updated": "unknown"}
        elif isinstance(entry, dict) and isinstance(entry.get("value"), str):
            parsed[str(name)] = {
                "value": entry["value"],
                "updated": str(entry.get("updated", "unknown")),
            }
        else:
            raise ConfigError(f"{path}: the entry for {name!r} has no string value.")
    return parsed


# -- resolution --------------------------------------------------------------


def resolve(name: str) -> str | None:
    """The one resolver: environment first, then the vault.

    Environment wins so a systemd unit, a container, or a one-off shell can
    override without touching disk — and so that every config written before
    the vault existed keeps behaving exactly as it did. The vault is the
    fallback that makes `kasa run` work with nothing exported at all.
    """
    if value := os.environ.get(name):
        return value
    return load_vault().get(name)


_cache: tuple[Path, tuple[int, int, int, int], Vault] | None = None


def load_vault(path: Path | None = None) -> Vault:
    """`Vault.load`, memoized on the file's mtime and size.

    `resolve` is called from provider construction, git operations and the
    redactor, so the uncached version would re-read and re-parse the file
    several times per turn. Keyed on the stat rather than merely on the path,
    so a `kasa vault set` in another terminal is picked up by a running daemon
    instead of being masked until restart.
    """
    global _cache
    target = path or vault_path()
    stamp = _stamp(target)
    if _cache is not None and _cache[0] == target and _cache[1] == stamp:
        return _cache[2]
    vault = Vault.load(target)
    _cache = (target, stamp, vault)
    return vault


def clear_cache() -> None:
    global _cache
    _cache = None


def _stamp(path: Path) -> tuple[int, int, int, int]:
    try:
        stat = path.stat()
    except OSError:
        return (-1, -1, -1, -1)
    try:
        directory_mode = path.parent.stat().st_mode & 0o777
    except OSError:
        directory_mode = -1
    return (stat.st_mtime_ns, stat.st_size, stat.st_mode & 0o777, directory_mode)


# -- placement ---------------------------------------------------------------


def check_placement(path: Path, *, clone_path: Path | None) -> None:
    """Refuse a vault that lives inside the long-term memory repo.

    Everything in that directory is committed and pushed by jobs that run
    unattended. A vault there is not at risk of leaking — it is scheduled to.
    """
    if clone_path is None:
        return
    resolved = path.expanduser().resolve()
    clone = clone_path.expanduser().resolve()
    if resolved == clone or clone in resolved.parents:
        raise ConfigError(
            INSIDE_MEMORY_REPO.format(
                path=resolved, clone=clone, default=vault_path(), env=VAULT_ENV
            )
        )


def enclosing_git_repo(path: Path) -> Path | None:
    """The nearest ancestor that is a git work tree, if there is one.

    Weaker than `check_placement` and reported as a warning rather than a
    refusal, because the honest answer here is "probably fine, but look": a
    home directory tracked as a dotfiles repo is the common way a `0600` file
    ends up on a remote anyway, and it is not something Kasa can tell apart
    from a deliberate arrangement.
    """
    resolved = path.expanduser().resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / ".git").exists():
            return candidate
    return None
