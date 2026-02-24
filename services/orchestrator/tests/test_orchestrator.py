from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.storage import DB_PATH


def _client() -> TestClient:
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()
    return TestClient(app)


def test_healthcheck() -> None:
    with _client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'ok'


def test_mnist_imagegen_end_to_end_returns_ranked_report() -> None:
    with _client() as client:
        project_response = client.post(
            '/api/projects',
            json={
                'name': 'MNIST tuning',
                'dataset_handle': 'mnist://train',
                'seed_question': 'Which configuration gives the best quality/stability tradeoff?',
            },
        )
        assert project_response.status_code == 200
        project_id = project_response.json()['project']['id']

        run_response = client.post(f'/api/projects/{project_id}/runs')
        assert run_response.status_code == 200
        run_id = run_response.json()['run']['id']

        execute_response = client.post(
            f'/api/runs/{run_id}/mnist-imagegen/execute',
            json={
                'objective': 'Find the most robust and high-quality MNIST image-generation settings',
                'candidates': [
                    {
                        'noise_schedule': 'linear',
                        'sampler': 'ddpm',
                        'guidance_scale': 2.0,
                        'learning_rate': 0.001,
                        'ema_decay': 0.97,
                        'grad_clip': 1.0,
                    },
                    {
                        'noise_schedule': 'cosine',
                        'sampler': 'heun',
                        'guidance_scale': 4.5,
                        'learning_rate': 0.0008,
                        'ema_decay': 0.989,
                        'grad_clip': 1.0,
                    },
                ],
            },
        )
        assert execute_response.status_code == 200
        report = execute_response.json()['report']

        assert report['total_candidates'] == 2
        assert report['evaluated_candidates'] == 2
        assert report['best']['rank'] == 1
        assert len(report['rankings']) == 2
        assert report['rankings'][0]['score'] >= report['rankings'][1]['score']

        timeline_response = client.get(f'/api/runs/{run_id}/timeline')
        assert timeline_response.status_code == 200
        timeline_events = timeline_response.json()['events']
        assert any('discovery phase' in event['message'] for event in timeline_events)
        assert any('MNIST image-generation batch completed' in event['message'] for event in timeline_events)


def test_duplicate_candidates_are_deduplicated_with_caveat() -> None:
    with _client() as client:
        project_id = client.post('/api/projects', json={'name': 'dupes', 'dataset_handle': 'mnist://train'}).json()['project']['id']
        run_id = client.post(f'/api/projects/{project_id}/runs').json()['run']['id']

        payload = {
            'objective': 'Find best MNIST settings while handling duplicate candidate rows robustly',
            'candidates': [
                {
                    'noise_schedule': 'cosine',
                    'sampler': 'ddim',
                    'guidance_scale': 4.5,
                    'learning_rate': 0.0008,
                    'ema_decay': 0.99,
                    'grad_clip': 1.0,
                },
                {
                    'noise_schedule': 'cosine',
                    'sampler': 'ddim',
                    'guidance_scale': 4.5,
                    'learning_rate': 0.0008,
                    'ema_decay': 0.99,
                    'grad_clip': 1.0,
                },
            ],
        }
        response = client.post(f'/api/runs/{run_id}/mnist-imagegen/execute', json=payload)
        assert response.status_code == 200
        report = response.json()['report']
        assert report['rejected_candidates'] == 1
        assert report['evaluated_candidates'] == 1
        assert any('duplicate' in caveat for caveat in report['caveats'])


def test_invalid_candidate_boundaries_fail_validation() -> None:
    with _client() as client:
        project_id = client.post('/api/projects', json={'name': 'bad-range', 'dataset_handle': 'mnist://train'}).json()['project']['id']
        run_id = client.post(f'/api/projects/{project_id}/runs').json()['run']['id']

        response = client.post(
            f'/api/runs/{run_id}/mnist-imagegen/execute',
            json={
                'objective': 'Find best settings',
                'candidates': [
                    {
                        'noise_schedule': 'linear',
                        'sampler': 'ddpm',
                        'guidance_scale': 20,
                        'learning_rate': 0.001,
                        'ema_decay': 0.97,
                        'grad_clip': 1.0,
                    }
                ],
            },
        )

        assert response.status_code == 422
