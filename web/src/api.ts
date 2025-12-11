const API_URL = 'http://localhost:8000';

export async function checkHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (!response.ok) throw new Error('Network response was not ok');
        return await response.json();
    } catch (error) {
        console.error('Health check failed:', error);
        return { status: 'offline', model_loaded: false };
    }
}
