from itertools import product
from bokeh.plotting import figure, show
from bokeh.models import OpenURL, Callback, TapTool
from bokeh.models import ColumnDataSource
import json
from neuroglancer_interface.modules.mfish_url import(
    create_mfish_url)

with open('../data/mouse1_gene_list.json', 'rb') as in_file:
    gene_data = json.load(in_file)

data = dict()
data['x_values'] = []
data['y_values'] = []
data['genes'] = []
data['grid_color'] = []
data['url'] = []
ic = 0
color_lookup = ['#aaaaaa', '#ffff00']
for x, y in product(
        ('cat', 'dog', 'frog'),
        ('alice', 'bob', 'charlie')):
    data['x_values'].append(x)
    data['y_values'].append(y)
    data['grid_color'].append(color_lookup[ic%2])
    ic += 1

for ii in range(len(data['x_values'])):
    data['genes'].append(gene_data[ii])
    url = create_mfish_url(
            mfish_bucket='mouse1-mfish-prototype',
            genes=[gene_data[ii]],
            colors=['green'],
            range_max=[10.0])
    #url = "https://bit.ly/3TlSKOl"
    data['url'].append(url)
    print(url)
    #print("")

x_range = list(set(data['x_values']))
x_range.sort()
y_range = list(set(data['y_values']))
y_range.sort()

#print(data)

data = ColumnDataSource(data=data)

p = figure(
        title='silly',
        x_range=x_range,
        y_range=y_range,
        tools=['hover', 'tap'],
        tooltips=[('name', '@y_values'),
                  ('pet', '@x_values'),
                  ('gene', '@genes')]
        )

p.rect(
    'x_values',
    'y_values',
    width=0.9,
    height=0.9,
    color='grid_color',
    source=data,
    hover_line_color='red')

taptool = p.select(type=TapTool)
taptool.callback = OpenURL(url='@url')

show(p)

