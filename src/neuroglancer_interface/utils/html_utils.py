import dominate
import dominate.tags

import pathlib


def write_basic_table(
        output_path=None,
        title=None,
        key_to_link=None,
        key_order=None,
        div_name=None,
        search_by=None,
        key_to_other_cols=None):
    """
    div_name is the name of the main div in the html


    search_by is a list of columns that can be searched by
    (will always include cls_name)

    key_to_other_cols maps same key as key_to_link to
    a dict with keys 'names', 'values' which point to
    ordered lists of data to be displayed (optional)
    """

    if search_by is None:
        search_by = []

    if key_order is not None:
        key_list = key_order
    else:
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
        with dominate.tags.table() as this_table:

            with this_table.add(dominate.tags.thead()) as this_table_header:
                header_row = dominate.tags.tr()
                for name in key_to_other_cols[key_list[0]]['names']:
                    header_row += dominate.tags.th(dominate.tags.a(name))
                header_row += dominate.tags.th(dominate.tags.a('URL'))
                this_table_header += header_row

            with this_table.add(dominate.tags.tbody(cls="list")) as this_table_body:

                for key_name in key_list:
                    this_url = key_to_link[key_name]
                    this_row = dominate.tags.tr()

                    if key_to_other_cols is not None:
                        this_data = key_to_other_cols[key_name]
                        for colname, colval in zip(this_data['names'], this_data['values']):
                            this_row += dominate.tags.td(dominate.tags.a(colval),
                                                         cls=colname)

                    this_row += dominate.tags.td(dominate.tags.a('link',
                                                                 href=this_url))

                    this_table_body += this_row

        doc += this_div
    doc.body += dominate.tags.script(src="https://cdnjs.cloudflare.com/ajax/libs/list.js/2.3.1/list.min.js")

    jcode = \
    '''
    var options = {
    '''
    jcode += f"valueNames: {search_by}"
    jcode += \
    '''
    };
    '''

    jcode += f"var userList = new List({div_name}, options);"

    doc.body += dominate.tags.script(jcode)

    with open(output_path, 'w') as out_file:
        out_file.write(doc.render())
