# Repository instructions

## Issue workflow

- Work on every issue in a dedicated Git worktree. Do not implement issue work
  directly in the primary checkout.
- Create all worktrees inside the repository under `.worktree/`, using one
  worktree and one branch per issue.
- When working on multiple issues, handle them sequentially. For each issue:
  implement and validate the change, open a pull request, and merge that pull
  request before starting work on the next issue.
- Do not combine multiple issues into one pull request.
