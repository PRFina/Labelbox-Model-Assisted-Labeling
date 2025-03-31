import sys
import labelbox as lb

API_KEY = None


if __name__ == "__main__":
    API_KEY = sys.argv[1] if len(sys.argv) > 1 else API_KEY
    if not API_KEY:
        raise ValueError("You need to provide the labelbox api key (with admin role)!")

    client = lb.Client(API_KEY)
    input("DANGER: this action will remove all projects, datasets, ontologies and features. Press enter to continue")

    for project in client.get_projects():
        project.delete()

    for dataset in client.get_datasets():
        dataset.delete()

    for ontology in client.get_unused_ontologies():
        client.delete_unused_ontology(ontology)

    from labelbox import LabelboxError

    for f in client.get_unused_feature_schemas():
        try:
            client.delete_unused_feature_schema(f)
        except LabelboxError as e:
            print(e)