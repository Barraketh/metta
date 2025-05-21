export async function getPolicyEvals(metric: string) {
  const response = await fetch(`http://localhost:8000/api/policy-evals?metric=${encodeURIComponent(metric)}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  const matrix = await response.json();
  return matrix;
}

export async function getMetrics() {
  const response = await fetch('http://localhost:8000/api/metrics');
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
}