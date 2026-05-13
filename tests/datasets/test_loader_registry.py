from iltools.datasets.loaders import (
    DatasetLoaderSpec,
    get_dataset_loader_spec,
    load_dataset_loader,
    register_dataset_loader,
    registered_dataset_loaders,
)


def test_registered_loaders_do_not_import_optional_backends_on_listing():
    names = registered_dataset_loaders()

    assert "lafan1_csv" in names
    assert "lerobot" in names
    assert "loco_mujoco" in names


def test_load_lafan1_csv_loader():
    loader_cls = load_dataset_loader("lafan1-csv")

    assert loader_cls.__name__ == "Lafan1CsvLoader"


def test_load_lerobot_loader():
    loader_cls = load_dataset_loader("lerobot")

    assert loader_cls.__name__ == "LeRobotLoader"


def test_register_dataset_loader_import_target():
    register_dataset_loader(
        "base_loader",
        module="iltools.datasets.base_loader",
        class_name="BaseLoader",
    )

    spec = get_dataset_loader_spec("base-loader")
    assert spec == DatasetLoaderSpec(
        module="iltools.datasets.base_loader",
        class_name="BaseLoader",
    )
    assert load_dataset_loader("base_loader").__name__ == "BaseLoader"
