const MOCK_MODE = process.env.MOCK_MODE === 'true' || process.env.NODE_ENV === 'development';

function isMockMode() {
    return MOCK_MODE;
}

function getModeInfo() {
    return {
        mode: MOCK_MODE ? 'mock-data' : 'production',
        isMock: MOCK_MODE,
        timestamp: new Date().toISOString()
    };
}

module.exports = {
    isMockMode,
    getModeInfo
};
