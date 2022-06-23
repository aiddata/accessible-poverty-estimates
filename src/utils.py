import sqlite3

from prefect import Client
from prefect.run_configs import LocalRun


class SpatiaLite():

    def __init__(self, sqlite_path, sqlite_lib_path):
        self.sqlite_path = sqlite_path
        self.sqlite_lib_path = sqlite_lib_path

        conn = self.build_connection()

        # initialise spatial table support
        # conn.execute('SELECT InitSpatialMetadata(1)')

        # this statement creates missing system tables,
        # including knn2, which we will use
        conn.execute('SELECT CreateMissingSystemTables(1)')

        conn.execute('SELECT EnableGpkgAmphibiousMode()')
        conn.close()


    def build_connection(self):

        # create connection to SQLite database
        conn = sqlite3.connect(self.sqlite_path)

        # allow SQLite to load extensions
        conn.enable_load_extension(True)

        # load SpatiaLite extension
        # see README.md for more information
        conn.load_extension(self.sqlite_lib_path)

        return conn


def run_flow(flow, executor, prefect_cloud_enabled, project_name):

    # flow.run_config = LocalRun()
    flow.executor = executor

    if prefect_cloud_enabled:
        flow_id = flow.register(project_name=project_name)
        client = Client()
        run_id = client.create_flow_run(flow_id=flow_id)
        state = run_id
    else:
        state = flow.run()

    return state

