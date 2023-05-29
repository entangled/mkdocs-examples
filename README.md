# Entangled Examples for MkDocs

[![Entangled badge](https://img.shields.io/badge/entangled-Use%20the%20source!-%2300aeff)](https://entangled.github.io/)
[![Deploy Pages](https://github.com/entangled/mkdocs-examples/actions/workflows/deploy-pages.yaml/badge.svg)](https://github.com/entangled/mkdocs-examples/actions/workflows/deploy-pages.yaml)

Here you find several examples of using Entangled with MkDocs.

## Literate Programming

This project is written using literate programming with [Entangled](https://entangled.github.io/). This means that some or all of its source code is contained in Markdown code blocks. These code blocks are kept synchronized with a compilable and debuggable version of the program. When you edit either the Markdown or the generated source files, make sure you have the Entangled daemon running:

```shell
entangled watch
```

Te generate a rendered version of the Markdown, you may run

```shell
mkdocs build
```

or if you want to do both:

```shell
mkdocs serve
```

Entangled will run inside the MkDocs event loop.

<details><summary>How to write code using Entangled</summary>

## Writing code

For didactic reasons we don’t always give the listing of an entire source file in one go. In stead, we use a system of references known as noweb (Ramsey 1994). You write code inside Markdown code blocks with the following syntax:

~~~markdown
``` {.cpp title="src/main.cpp"}
#include <cstdlib>
#include <iostream>

<<main-function>>
```
~~~

This creates a file `src/main.cpp` containing a not-yet-specified `main` function. This main function will print a friendly message on the screen.

~~~markdown
``` {.cpp title="#hello-world"}
std::cout << "Hello, World!" << std::endl;
```
~~~

To complete the program we need to create the `main` function.

~~~markdown
``` {.cpp title="#main-function"}
int main(int argc, char **argv) {
    <<hello-world>>
}
```
~~~

Code blocks can be appended on with more code by repeating the same name for the code block.

~~~markdown
``` {.cpp title="#hello-world"}
return EXIT_SUCCESS;
```
~~~
</details>

## MkDocs

The documentation is generated using [MkDocs Material](https://squidfunk.github.io/mkdocs-material/). See those pages for information on supported syntax and available plugins.

You may install the dependencies using `poetry` and activate the virtual environment before building:

```
poetry install
poetry shell
mkdocs build
```

If you don't (want to) use `poetry`:

```
python -m venv venv
./venv/bin/activate
pip install .
mkdocs build
```

## License

This template is licensed under the [Apache License v2.0](https://www.apache.org/licenses/LICENSE-2.0). See `LICENSE`.
