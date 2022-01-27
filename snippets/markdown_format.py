from glob import glob

import isort
from black import FileMode, format_str


def main():
    CODE_PATH = r"E:\repositorio\github\ml-codebase\snippets"
    markdown_name = "Receipts.md"
    files = [
        file
        for file in glob(CODE_PATH + "\*", recursive=True)
        + glob(CODE_PATH + "\*\*", recursive=True)
        if ".py" in file
    ]
    code_list = [segment_file(file) for file in files]
    code_list = sorted(code_list, key=lambda k: k["segment"] + ":" + k["file"])
    for key, code in enumerate(code_list):
        if key == 0:
            save_to_markdown(code, markdown_name, "w+")
        else:
            save_to_markdown(code, markdown_name, "a")


def segment_file(file):
    file_list = file.split("\\")
    segment = file_list[len(file_list) - 2]
    file_name = file_list[len(file_list) - 1]

    with open(file, "r") as fopened:
        code = fopened.read()

    return {"segment": segment, "file": file_name, "code": code}


def save_to_markdown(data, markdown_name="teste.md", mode="w+"):
    segment = data["segment"]
    file = data["file"]
    code = data["code"]

    with open(markdown_name, mode) as fopen:
        try:
            code_formated = format_str(code, mode=FileMode())
        except:
            code_formated = code

        try:
            code_formated = isort.code(code_formated)
        except:
            code_formated = code_formated

        form = f"\n## {segment}/{file}\n```python\n{code_formated}```"
        fopen.write(form)


if __name__ == "__main__":
    main()
