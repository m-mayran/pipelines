# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import unittest

from _pytest.monkeypatch import MonkeyPatch
from googleapiclient import discovery
from googleapiclient.http import HttpMock
from google_cloud_pipeline_components.experimental.notebooks.executor import NotebooksExecutor  # pylint: disable=relative-beyond-top-level,line-too-long
from kfp import components
import pytest
from types import SimpleNamespace

ROOT_DIR = os.path.abspath('.')
COMPONENT_DIR = os.path.join(ROOT_DIR, 'google_cloud_pipeline_components', 'experimental', 'notebooks')
DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')


def componentfile():
  return os.path.join(COMPONENT_DIR, 'component.yaml')


def datafile(filename):
  return os.path.join(DATA_DIR, filename)


@pytest.fixture(scope='class')
def monkeypatch_for_class(request):
  request.cls.monkeypatch = MonkeyPatch()


@pytest.mark.usefixtures('monkeypatch_for_class')
class TestNotebookExecutor():

  def monkey_make_client_notebooks(self, http):
    http = HttpMock(datafile('notebooks.json'), {'status': '200'})
    return discovery.build('notebooks', 'v1', http=http, cache_discovery=False)

  def monkey_make_client_ml(self, http):
    http = HttpMock(datafile('ml.json'), {'status': '200'})
    return discovery.build('ml', 'v1', http=http, cache_discovery=False)

  def test_cmd(self):
    # Loads component from yaml file.
    en_component = components.load_component_from_file(componentfile())
    en_component_op = en_component(
        project_id='mock-project',
        input_notebook_file='gs://mock-input-notebook-file.ipynb',
        output_notebook_folder='gs://mock-output-notebook-folder',
        execution_id='mock_execution_id')
    # print(en_component_op.__dict__)

    # Creates command line arguments from yaml data
    arguments = en_component_op.arguments
    keys = [f'--{k}' for k in arguments.keys()]
    values = arguments.values()

    args_calculated = [item for pair in zip(keys, values) for item in pair]  # pylint: disable=g-complex-comprehension
    args_expected = [
        '--project_id', 'mock-project',
        '--input_notebook_file', 'gs://mock-input-notebook-file.ipynb',
        '--output_notebook_folder', 'gs://mock-output-notebook-folder',
        '--execution_id', 'mock_execution_id',
    ]

    assert len(args_calculated) == len(args_expected)
    assert all([a == b for a, b in zip(args_calculated, args_expected)])

  @pytest.mark.parametrize(
      'input_notebook_file,output_notebook_folder', [
          ('mock-input-notebook-file.ipynb', 'mock-output-notebook-folder'),
          ('gs://mock-input-notebook-file.ipynb',
           'gs://mock-output-notebook-folder'),
      ])
  def test_body_execution_create(
      self, input_notebook_file, output_notebook_folder):
    mock_inputs = SimpleNamespace(
        project_id='mock-project',
        input_notebook_file=input_notebook_file,
        output_notebook_folder=output_notebook_folder,
        execution_id='mock_execution_id',
        location='us-central1',
        master_type='n1-standard-4',
        scale_tier='CUSTOM',
        accelerator_type=None,
        accelerator_core_count='0',
        labels='src=notebooks_executor_api',
        container_image_uri='gcr.io/deeplearning-platform-release/base-cpu:latest',
        params_yaml_file=None,
        parameters=None,
        block_pipeline=True,
        fail_pipeline=True
    )

    body = NotebooksExecutor._build_body_executor(mock_inputs)
    expected_body_file = open(datafile('expected_body.json'))
    expected_body = json.load(expected_body_file)
    unittest.TestCase().assertDictEqual(body, expected_body)

  @pytest.mark.parametrize(
      'block_pipeline,output_state', [
          (False, 'PREPARING'),
          (True, 'SUCCEEDED'),
      ])
  def test_block_pipeline(self, block_pipeline, output_state):

    def mock_execute_executions_create(*args):
      http = HttpMock(datafile('executions_create_oo.json'), {'status': '200'})
      request = args[0]
      response = request.execute(http=http)
      return response, None

    def mock_wait_operations_get(*args):
      http = HttpMock(
          datafile('operations_get_done_preparing.json'), {'status': '200'})
      request = args[0]
      response = request.execute(http=http)
      return response

    def mock_wait_executions_get(*args):
      http = HttpMock(
          datafile('executions_get_succeeded.json'), {'status': '200'})
      request = args[0]
      response = request.execute(http=http)
      return response

    self.monkeypatch.setattr(
        NotebooksExecutor, '_make_client_notebooks', self.monkey_make_client_notebooks)
    self.monkeypatch.setattr(
        NotebooksExecutor, '_make_client_ml', self.monkey_make_client_ml)
    self.monkeypatch.setattr(
        NotebooksExecutor, '_execute_executions_create', mock_execute_executions_create)
    self.monkeypatch.setattr(
        NotebooksExecutor, '_wait_operations_get', mock_wait_operations_get)
    self.monkeypatch.setattr(
        NotebooksExecutor, '_wait_executions_get', mock_wait_executions_get)

    outputs = NotebooksExecutor.execute_notebook(
        project_id='mock-project',
        input_notebook_file='mock-input-notebook-file.ipynb',
        output_notebook_folder='mock-output-notebook-folder',
        execution_id='mock_execution_id',
        block_pipeline=block_pipeline
    )

    assert outputs[0] == output_state

  @pytest.mark.parametrize('fail_pipeline', [(False), (True)])
  def test_handle_error(self, fail_pipeline):
    if fail_pipeline:
      with pytest.raises(RuntimeError):
        _, _ = NotebooksExecutor._handle_error(
            error_message='mock-error', fail_pipeline=fail_pipeline)
    else:
      _, err = NotebooksExecutor._handle_error(
          error_message='mock-error', fail_pipeline=fail_pipeline)
      assert err == 'mock-error'
