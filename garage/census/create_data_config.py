import json
import pathlib


def main():
    parent_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/lydian/mfish/mouse_3/20230327_registered/best.cl')
    assert parent_dir.is_dir()

    config_list = []
    for pth in [n for n in parent_dir.iterdir() if n.is_file()]:
        if not pth.name.endswith('.nii.gz'):
            print(f"skipping {pth}")
            continue
        tag = 'cluster_'+pth.name.split('.')[0]
        this = {'tag': tag, 'path': str(pth.resolve().absolute())}
        config_list.append(this)

    with open('data_config.json', 'w') as out_file:
        out_file.write(json.dumps(config_list, indent=2))

    print(f"{len(config_list)} configs")

if __name__ == "__main__":
    main()
