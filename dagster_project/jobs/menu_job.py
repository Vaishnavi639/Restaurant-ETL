from dagster import job
from dagster_project.assets.menu_assets import menu_etl_asset


@job
def menu_job():
    menu_etl_asset()

