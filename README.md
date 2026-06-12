# RLinf-kunlun-recipe

`RLinf-kunlun-recipe` hosts kunlun specified recipes based on [RLinf](https://github.com/KunlunxinAD/RLinf) contributed by the community.

## Usage

`RLinf-kunlun-recipe` can be used as a submodule of `RLinf`:

```bash
git clone https://github.com/KunlunxinAD/RLinf.git
git clone https://github.com/KunlunxinAD/RLinf-kunlun-recipe.git
cp -r RLinf-kunlun-recipe ./RLinf
cd RLinf
```

### Code Linting and Formatting

To maximize flexiblility but minimize meaningless changes, we apply `pre-commit` but only force code linting and formatting with `ruff`. Use it as follows:

```bash
pip install pre-commit
pre-commit install
# for staged changes
pre-commit run
# for all files in the repo
pre-commit run --all-files
```
