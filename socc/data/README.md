# SOCC Package Data

Data for easy testing of socc.

## Including package data

Modify your package's `pyproject.toml` file.
Update the [tool.setuptools.package_data](https://setuptools.pypa.io/en/latest/userguide/datafiles.html#package-data)
and point it at the correct files.
Paths are relative to `package_dir`.

Package data can be accessed at run time with `importlib.resources` or the `importlib_resources` back port.
See https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime
for suggestions.

If modules within your package will access internal data files using
[the recommended approach](https://setuptools.pypa.io/en/latest/userguide/datafiles.html#accessing-data-files-at-runtime),
you may need to include `importlib_resources` in your package dependencies.
In `pyproject.toml`, include the following in your `[project]` table.
```
dependencies = [
    "importlib-resources;python_version<'3.10'",
]
```

## Manifest

* `molecules.py` : A dictionary of molecular structures for common tests.
