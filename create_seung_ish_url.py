from url_utils import json_to_url, get_ish_image_layer
import json
import argparse


def get_base_url():
    #return "https://neuroglancer-demo.appspot.com/#!"
    return "https://neuromancer-seung-import.appspot.com/#!"


def get_url(
        bucket_name,
        img_name):

    state = dict()

    img_layer = get_ish_image_layer(
        bucket_name=bucket_name,
        img_name=img_name)


    state["layers"] = [img_layer]

    state["selectedLayer"] = {"layer": img_layer["name"],
                              "visible": True}

    state["layout"] = "xy"

    url = f"{get_base_url()}{json_to_url(json.dumps(state))}"
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_name",
                        type=str,
                        default="sfd-eastern-bucket")
    args = parser.parse_args()

    print(get_url(
            bucket_name=args.bucket_name,
            img_name="100047769_64"))

if __name__ == "__main__":
    main()
