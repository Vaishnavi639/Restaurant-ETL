from dagster import Definitions
from dagster_project.assets.menu_assets import menu_etl_asset
from dagster_project.jobs.menu_job import menu_job

defs = Definitions(
    assets=[menu_etl_asset],
    jobs=[menu_job],
)

