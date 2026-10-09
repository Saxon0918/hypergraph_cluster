# Data directory

The portable experiment inputs are stored under `data/raw/` and referenced by
repository-relative paths in `configs/local_*.yaml`.  Paths are resolved from the
project root, so the same checkout works on a workstation or server regardless of
its absolute location or the shell's current directory.

Downloaded archives under `data/raw/downloads/` are ignored because they duplicate
the extracted inputs. Generated caches belong in `data/processed/`; experiment
artifacts belong in `runs/`. Source identity is recorded in each run's
`sources_manifest.json`.

Dataset copyrights, licenses, and citation requirements remain with their original
providers. Verify redistribution permission before pushing these files to a public
repository.

