import json
import pathlib

def main():
    parent_dir = pathlib.Path(
        '/allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_638850/alignment/RegPrelimDefNN_mouse3/iter0')
    assert parent_dir.is_dir()

    structure_dir = parent_dir / 'structure_masks'
    structure_set_dir = parent_dir / 'structure_set_masks'

    config_list = []
    assert structure_dir.is_dir()
    structure_path_list = [n for n in structure_dir.iterdir() if n.is_file()]
    for pth in structure_path_list:
         if not pth.name.endswith('.nii.gz'):
              print(f"skipping {pth}")
              continue
         tag = pth.name.split('.')[0]
         this = {'path': str(pth.resolve().absolute()),
                 'structure': tag}
         config_list.append(this)

    assert structure_set_dir.is_dir()
    for pth in [n for n in structure_set_dir.iterdir() if n.is_file()]:
        if not pth.name.endswith('.nii.gz'):
            print(f"skipping {pth}")
            continue
        tag = pth.name.split('.')[0]
        this = {'path': str(pth.resolve().absolute()),
                'structure': tag}
        config_list.append(this)
    with open('mask_config.json', 'w') as out_file:
        out_file.write(json.dumps(config_list, indent=2))

if __name__ == "__main__":
    main()
