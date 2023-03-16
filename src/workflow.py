
from kfp import dsl
import mlrun


# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(name="iris-git-demo")
def pipeline(dataset_uri,model_name="iris_model"):
    train = mlrun.run_function(
        "trainer",
        inputs={"dataset": dataset_uri},
        params = {
            "model_class": "sklearn.ensemble.RandomForestClassifier",
            "train_test_split_size": 0.2,
            "label_columns": "label",
            "model_name": model_name,
        }, 
        handler='train',
        outputs=["model"],
    )

    # Deploy the trained model as a serverless function
    mlrun.deploy_function(
        "serving",
        models=[
            {
                "key": model_name,
                "model_path": train.outputs["model"],
                "class_name": 'mlrun.frameworks.sklearn.SklearnModelServer',
            }
        ],
    )
