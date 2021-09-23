from google.protobuf.duration_pb2 import Duration

from feast import Entity, Feature, FeatureView, FileSource, ValueType

user_categories = FileSource(
    path=r"E:\repositorio\development\feast\repo\data\user_categories.parquet",
    event_timestamp_column="event_timestamp",
)

user = Entity(name="user_id", value_type=ValueType.INT64, description="user id",)

user_caterogicals_view = FeatureView(
    name="user_categories",
    entities=["user_id"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="user_category", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=user_categories,
    tags={},
)

driver_categories = FileSource(
    path=r"E:\repositorio\development\feast\repo\data\driver_categories.parquet",
    event_timestamp_column="event_timestamp",
)

driver = Entity(name="driver_id", value_type=ValueType.INT64, description="user id",)

driver_caterogicals_view = FeatureView(
    name="driver_categories",
    entities=["driver_id"],
    ttl=Duration(seconds=86400 * 1),
    features=[
        Feature(name="driver_category", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=driver_categories,
    tags={},
)
