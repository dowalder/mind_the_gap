#!/usr/bin/env python3

import argparse
import pathlib


HTML_BODY = """<!DOCTYPE html> 
<html>
<head>
    <title>Neural Style Transfer Summary</title>
</head>
<body>
    <center>
        <h1>Neural Style Transfer Summary</h1>
    </center>

    <table border="5" bordercolor="black" align="center">
        {content}
    </table>
</body>
<html>"""

TABLE_ROW = """<tr>{content}</tr>"""
TABLE_HEADER = """<th>{content}</th>"""
TABLE_CELL = """<td>{content}</td>"""
IMG_CELL = """<td><img src={path} border=3 height=480 width=640></img></td>"""


def make_html(path: pathlib.Path, max_images=30):
    assert path.is_dir(), "Dir not found: {}".format(path)

    folder_names = []
    image_names = set()
    for folder in path.iterdir():
        if not folder.is_dir():
            continue
        image_names.update(map(lambda x: x.name, folder.glob("*.jpg")))
        folder_names.append(folder)
    image_names = list(image_names)
    image_names.sort()
    folder_names.sort()
    if len(image_names) > max_images:
        print("WARNING: restricting displayed images to {}".format(max_images))
        image_names = image_names[:max_images]

    table_content = []

    # table header
    table_content.append("<tr>" + TABLE_CELL.format(content=""))  # empty cell in the top left
    table_content.extend(map(lambda x: TABLE_HEADER.format(content=x), image_names))  # image names in the top row

    # insert rows
    for folder in folder_names:
        table_content.append("<tr>")
        table_content.append(TABLE_CELL.format(content=folder.name))
        # there is no guarantee that the image path is valid (unless the folders contain only images of the same name)
        table_content.extend(map(lambda x: IMG_CELL.format(path=folder.name + "/" + x), image_names))
        table_content.append("</tr>")

    html_text = HTML_BODY.format(content="".join(table_content))

    (path / "index.html").write_text(html_text)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", required=True, help="path pointing to directory containing several images")

    args = parser.parse_args()

    path = pathlib.Path(args.path)

    make_html(path)


if __name__ == "__main__":
    main()