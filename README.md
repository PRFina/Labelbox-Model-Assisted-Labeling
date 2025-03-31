Simple scripts to try Labelbox's MAL feature to import pre-annotated labels with video assets.

### Install
With pip, create a venv and run
```shell
pip install -r requirments.txt
```

### Run
With `pip`, activate the venv and run
```shell
python <script.py> <API KEY>
```
With `uv`:
```shell
uv run <script.py> <API KEY>
```

## Script Execution
* Create a new dataset with one  data row and video asset
* Create a simple ontology. Each script creates an ontology with a different tool type (classifcation, bbox, segmentation mask)
* Create a project and add the datarow to a batch
* Create a MAL payload. MAL annotations are randomly generated to simulate manual annotation process
* Upload MAL payload to the project
* Wait for user input since MAL annotations needs to be manually moved from *initial labeling* workflow step to *Initial review*. 
* Export the project annotations and dump in a `.ndjson` file.