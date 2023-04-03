import json
from neuroglancer_interface.census.census import (
    run_census)


def main():
    mask_config = json.load(open('mask_config.json', 'rb'))
    data_config = json.load(open('data_config.json', 'rb'))
    run_census(
        mask_config_list=mask_config,
        data_config_list=data_config,
        h5_path='/allen/aibs/technology/danielsf/mouse_638550_census_230403.h5',
        n_processors=6)

if __name__ == "__main__":
    main()
