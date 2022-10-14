import argparse

def get_base_url():
    return "https://neuroglancer-demo.appspot.com/#!%7B%22layers%22:"


def add_segmentation(
        base_url,
        segmentation_bucket,
        segmentation_name='segmentation'):
    """
    Add a segmentation layer to the base_url. Return URL with reference to the
    segmentation layer.

    Parameters
    ----------
    base_url: str
        The URL to which we are appending a segmentation layer

    segmentation_bucket: str
        The name of the S3 bucket containing the segmentation

    segmentation_name: str
        The name that will appear in the segmentation tab in neuroglancer
    """

    prefix = "%5B%7B%22type%22:%22segmentation%22%2C%22source%22:"
    prefix += "%22precomputed://"

    suffix = "%22%2C%22tab%22:%22source%22%2C%22name%22:"
    suffix += f"%22{segmentation_name.replace(' ','%20')}%22%7D%5D%2C%22"
    suffix += "selectedLayer%22:%7B%22visible%22:true%2C%22"
    suffix += "layer%22:%22new%20layer%22%7D%2C%22layout%22:%224panel%22%7D"

    new_url = f"{base_url}{prefix}s3://{segmentation_bucket}{suffix}"


    return new_url



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--segmentation_bucket',
                        type=str,
                        default='mouse1-atlas-prototype')
    parser.add_argument('--segmentation_name',
                        type=str,
                        default='segmentation')

    args = parser.parse_args()

    url = get_base_url()
    url = add_segmentation(
                base_url=url,
                segmentation_bucket=args.segmentation_bucket,
                segmentation_name=args.segmentation_name)

    print(url)


if __name__ == "__main__":
    main()
