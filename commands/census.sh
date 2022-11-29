nice python run_census.py --n_processors 6 \
--mask_dir /allen/programs/celltypes/workgroups/rnaseqanalysis/mFISH/michaelkunst/MERSCOPES/mouse/atlas/mouse_1/alignment/RegPrelimDefNN_mouse1/iter0/structure_set_masks \
--structure_lookup ../ontology_parcellation_tools/input/structure_sets.csv \
--celltypes_dir /allen/aibs/technology/danielsf/mouse1_cell_types/ \
--mfish_dir /allen/aibs/technology/danielsf/mouse1_ome_zarr/ --output_path /allen/aibs/technology/danielsf/mouse1_structure_sets_census.json

nice python run_census.py --n_processors 6 \
--structure_lookup ../ontology_parcellation_tools/downloaded_graphs/1_adult_mouse_brain_graph.json \
--celltypes_dir /allen/aibs/technology/danielsf/mouse1_cell_types/ \
--mfish_dir /allen/aibs/technology/danielsf/mouse1_ome_zarr/ --output_path /allen/aibs/technology/danielsf/mouse1_structures_census.json

