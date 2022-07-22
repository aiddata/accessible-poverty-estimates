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


from prefect import Flow
from prefect.tasks.prefect import create_flow_run, wait_for_flow_run

def run_flow(flow, executor, prefect_cloud_enabled=False, project_name=None, parent_flow_name=None):


    if prefect_cloud_enabled:
        client = Client()

        if isinstance(flow, list):
            if parent_flow_name is None:
                parent_flow_name == "parent-flow"

            for existing_flow in flow:
                existing_flow.executor = executor
                _ = existing_flow.register(project_name=project_name)

            with Flow(parent_flow_name) as parent_flow:

                for ix, existing_flow in enumerate(flow):

                    # assumes you have registered the following flows in a project named "examples"
                    sub_flow = create_flow_run(flow_name=existing_flow.name, project_name=project_name)

                    if ix > 0:
                        sub_flow.set_upstream(wait_for_flow)

                    # if ix < len(flow) - 1:
                    wait_for_flow = wait_for_flow_run(sub_flow, raise_final_state=True)

            parent_flow.executor = executor
            flow_id = parent_flow.register(project_name=project_name)
            run_id = client.create_flow_run(flow_id=flow_id)
            state = run_id

        else:
            flow.executor = executor
            flow_id = flow.register(project_name=project_name)
            run_id = client.create_flow_run(flow_id=flow_id)
            state = run_id

    else:
        if isinstance(flow, list):
            for ix, existing_flow in enumerate(flow):
                existing_flow.executor = executor
                state = existing_flow.run()


        else:
            flow.executor = executor
            state = flow.run()

    # # flow.run_config = LocalRun()
    # flow.executor = executor

    # if prefect_cloud_enabled:
    #     client = Client()

    #     flow_id = flow.register(project_name=project_name)

    #     run_id = client.create_flow_run(flow_id=flow_id)
    #     state = run_id
    # else:
    #     state = flow.run()

    return state

