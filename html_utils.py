import dominate
import dominate.tags

import pathlib


def write_basic_table(
        output_path=None,
        title=None,
        key_to_link=None,
        div_name=None,
        cls_name=None):

    key_list = list(key_to_link.keys())
    key_list.sort()

    doc = dominate.document(title=title)
    doc += dominate.tags.h1(title)
    doc.head += dominate.tags.link(
                    href="reconstruction_table.css",
                    rel="stylesheet",
                    type="text/css",
                    media="all")

    # Chrome was not loading changes, due its internal cache
    # https://stackoverflow.com/questions/34851243/how-to-make-index-html-not-to-cache-when-the-site-contents-are-changes-in-angula
    # https://stackoverflow.com/questions/46535832/how-to-declare-metadata-tags-using-python-dominate
    doc.head += dominate.tags.meta(http_equiv="Cache-control",
                              content="no-cache, no-store, must-revalidate")
    doc.head += dominate.tags.meta(http_equiv="Pragma",
                              content="no-cache")

    with dominate.tags.div(id=div_name) as this_div:
        this_div += dominate.tags.input_(cls="search", placeholder="Search")
        with dominate.tags.table().add(dominate.tags.tbody(cls="list")) as this_table:

            for key_name in key_list:
                this_url = key_to_link[key_name]
                this_row = dominate.tags.tr()
                this_row += dominate.tags.td(dominate.tags.a(key_name),
                                             cls=cls_name)
                this_row += dominate.tags.td(dominate.tags.a('link',
                                                             href=this_url))
                this_table += this_row

        doc += this_div
    doc.body += dominate.tags.script(src="https://cdnjs.cloudflare.com/ajax/libs/list.js/1.5.0/list.min.js")

    jcode = \
    '''
    var options = {
    '''
    jcode += f"valueNames: [ '{cls_name}', ]"
    jcode += \
    '''
    };
    '''

    jcode += f"var userList = new List({div_name}, options);"

    doc.body += dominate.tags.script(jcode)

    with open(output_path, 'w') as out_file:
        out_file.write(doc.render())
