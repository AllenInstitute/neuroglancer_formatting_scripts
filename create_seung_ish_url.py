from url_utils import json_to_url, get_image_layer
import json

def get_base_url():
    #return "https://neuroglancer-demo.appspot.com/#!"
    return "https://neuromancer-seung-import.appspot.com/#!"

def get_url():

    img_layer = get_image_layer(
        bucket_name="sfd-eastern-bucket",
        dataset_name="eg.nii.gz",
        public_name="example",
        color="red",
        range_max=10.0)

    img_layer["source"] = img_layer["source"].replace('zarr', 'nifti')
    img_layer["source"] = "nifti://https://sfd-eastern-bucket.s3.amazonaws.com/eg.nii.gz"
    img_layer.pop("shader")

    img_layer = dict()
    img_layer["source"] = "nifti://https://sfd-eastern-bucket.s3.amazonaws.com/eg.nii.gz"
    img_layer["type"] = "image"
    img_layer["blend"] = "default"
    img_layer["shaderControls"] = {}
    img_layer["name"] = "eg.nii.gz"

    layers = dict()
    layers["dimensions"] = {"x": [1.0e-5, "m"],
                            "y": [1.0e-5, "m"],
                            "z": [0.0001, "m"]}
    layers["crossSectionScale"] = 2.6
    layers["projectionScale"] = 2048
    layers["layers"] = [img_layer]
    layers["selectedLayer"] = {"visible": True, "layer": "new layer"}
    layers["layout"] = "4panel"

    layers = dict()
    layers["layers"] = [img_layer]
    navigation = {"pose": {
      "position": {
        "voxelSize": [
          9971.21875,
          100002.609375,
          10002.6123046875
        ],
        "voxelCoordinates": [
          0.5247101783752441,
          -12.514060974121094,
          136.0876007080078
        ]
      }
    }, "zoomFactor": 9971.22}
    layers["navigation"] = navigation
    layers["layout"] = "4panel"


    state = {
  "layers": [
    {
      "source": "nifti://https://sfd-eastern-bucket.s3.amazonaws.com/eg.nii.gz",
      "type": "image",
      "blend": "default",
      "shaderControls": {},
      "name": "eg.nii.gz"
    }
  ],
  "navigation": {
    "pose": {
      "position": {
        "voxelSize": [
          9971.21875,
          100002.609375,
          10002.6123046875
        ],
        "voxelCoordinates": [
          0.5247101783752441,
          -12.514060974121094,
          136.0876007080078
        ]
      }
    },
    "zoomFactor": 9971.21875
  },
  "layout": "4panel"
}
    url = f"{get_base_url()}{json_to_url(json.dumps(state))}"
    return url

def main():
    print(get_url())

if __name__ == "__main__":
    main()
