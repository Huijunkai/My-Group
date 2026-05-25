const axios = require('axios');
const FormData = require('form-data');
const { DEFAULT_HEADERS } = require('./constants');

const GUILIN_SERVER_1 = 'http://221.7.150.22:10005';
const GUILIN_SERVER_2 = 'http://221.7.150.20:10004';

const MAX_RETRIES = 3;
const RETRY_DELAY_BASE = 1000;

async function retryRequest(fn, operationName) {
    let lastError;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            const isRetryable = error.code === 'ECONNRESET' || error.code === 'ETIMEDOUT' ||
                error.code === 'ECONNREFUSED' || (error.response && error.response.status >= 500);
            
            if (!isRetryable || attempt === MAX_RETRIES) {
                throw error;
            }
            
            const delay = RETRY_DELAY_BASE * Math.pow(2, attempt - 1);
            console.warn(`${operationName} 第${attempt}次请求失败 (${error.message})，${delay}ms后重试...`);
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
    throw lastError;
}

const GUILIN_BUILDING_SERVER_MAP = {
    '4320': GUILIN_SERVER_1,
    '4509': GUILIN_SERVER_1,
    '4722': GUILIN_SERVER_1,
    '4812': GUILIN_SERVER_1,
    '6436': GUILIN_SERVER_1,
    '6819': GUILIN_SERVER_1,
    'B101': GUILIN_SERVER_2,
    'B102': GUILIN_SERVER_2,
    'B8': GUILIN_SERVER_2
};

const GUILIN_BUILDINGS_SERVER_1 = [
    { xiaoqu_id: 'glxq', loudong_id: '4320', loudong_name: '桂林校区9栋', xiaoqu_name: '桂林校区' },
    { xiaoqu_id: 'glxq', loudong_id: '4509', loudong_name: '桂林校区7栋', xiaoqu_name: '桂林校区' },
    { xiaoqu_id: 'glxq', loudong_id: '4722', loudong_name: '桂林校区12栋', xiaoqu_name: '桂林校区' },
    { xiaoqu_id: 'glxq', loudong_id: '4812', loudong_name: '桂林校区13栋', xiaoqu_name: '桂林校区' },
    { xiaoqu_id: 'glxq', loudong_id: '6436', loudong_name: '桂林校区14A栋', xiaoqu_name: '桂林校区' },
    { xiaoqu_id: 'glxq', loudong_id: '6819', loudong_name: '桂林校区14B栋', xiaoqu_name: '桂林校区' }
];

const GUILIN_BUILDINGS_SERVER_2 = [
    { xiaoqu_id: 'glxq', loudong_id: 'B101', loudong_name: '桂林校区10A号楼', xiaoqu_name: '桂林校区' },
    { xiaoqu_id: 'glxq', loudong_id: 'B102', loudong_name: '桂林校区10B号楼', xiaoqu_name: '桂林校区' },
    { xiaoqu_id: 'glxq', loudong_id: 'B8', loudong_name: '桂林校区8号楼', xiaoqu_name: '桂林校区' }
];

function getServerForBuilding(buildingId) {
    if (GUILIN_BUILDING_SERVER_MAP[buildingId]) {
        return GUILIN_BUILDING_SERVER_MAP[buildingId];
    }
    return GUILIN_SERVER_1;
}

function getServerForRoom(roomId) {
    if (!roomId) return GUILIN_SERVER_1;
    
    for (const [buildingId, server] of Object.entries(GUILIN_BUILDING_SERVER_MAP)) {
        if (roomId.includes(buildingId) || roomId.startsWith('H' + buildingId)) {
            return server;
        }
    }
    
    if (roomId.startsWith('H') && roomId.length === 5) {
        return GUILIN_SERVER_1;
    }
    if (roomId.startsWith('H') && roomId.length === 4) {
        return GUILIN_SERVER_2;
    }
    
    return GUILIN_SERVER_1;
}

async function getGuilinBuildings(accessToken) {
    const headers = {
        ...DEFAULT_HEADERS,
        'Authorization': `bearer ${accessToken}`,
        'Content-Type': 'multipart/form-data'
    };

    const allBuildings = [];

    try {
        const formData1 = new FormData();
        formData1.append('areaId', 'glxq');

        const response1 = await retryRequest(async () => {
            return await axios.post(`${GUILIN_SERVER_1}/v1/cgElec/loudong/query`, formData1, {
                headers,
                timeout: 15000
            });
        }, '桂林校区一号服务器获取楼栋');

        if (response1.data && response1.data.data) {
            allBuildings.push(...response1.data.data);
        }
    } catch (error) {
        console.error('桂林校区一号服务器获取楼栋失败:', error.message);
        allBuildings.push(...GUILIN_BUILDINGS_SERVER_1);
    }

    try {
        const formData2 = new FormData();
        formData2.append('areaId', 'glxq');

        const response2 = await retryRequest(async () => {
            return await axios.post(`${GUILIN_SERVER_2}/v1/cgElec/loudong/query`, formData2, {
                headers,
                timeout: 15000
            });
        }, '桂林校区二号服务器获取楼栋');

        if (response2.data && response2.data.data) {
            allBuildings.push(...response2.data.data);
        }
    } catch (error) {
        console.error('桂林校区二号服务器获取楼栋失败:', error.message);
        allBuildings.push(...GUILIN_BUILDINGS_SERVER_2);
    }

    const uniqueBuildings = [];
    const seenIds = new Set();
    for (const building of allBuildings) {
        if (!seenIds.has(building.loudong_id)) {
            seenIds.add(building.loudong_id);
            uniqueBuildings.push(building);
        }
    }

    return uniqueBuildings;
}

async function getGuilinRooms(accessToken, buildingId, page = 1, size = 100) {
    const serverUrl = getServerForBuilding(buildingId);
    const headers = {
        ...DEFAULT_HEADERS,
        'Authorization': `bearer ${accessToken}`,
        'Content-Type': 'multipart/form-data'
    };

    const formData = new FormData();
    formData.append('loudong_id', buildingId);
    formData.append('current', page);
    formData.append('size', size);

    try {
        const response = await retryRequest(async () => {
            return await axios.post(`${serverUrl}/v1/cgElec/room/query`, formData, {
                headers,
                timeout: 15000
            });
        }, `桂林校区获取房间 (楼栋: ${buildingId})`);

        if (response.data) {
            return {
                data: response.data.data || [],
                total: response.data.total || 0,
                pages: response.data.pages || 0,
                current: response.data.current || page
            };
        }
    } catch (error) {
        console.error(`桂林校区获取房间失败 (楼栋: ${buildingId}):`, error.message);
        if (error.response) {
            console.error('响应状态:', error.response.status);
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
    }

    return { data: [], total: 0, pages: 0, current: page };
}

async function getGuilinAllRoomsByBuilding(accessToken, buildingId) {
    const allRooms = [];
    let page = 1;
    const size = 100;
    
    while (true) {
        const result = await getGuilinRooms(accessToken, buildingId, page, size);
        if (result.data && result.data.length > 0) {
            allRooms.push(...result.data);
        }
        if (page >= result.pages || result.data.length === 0) {
            break;
        }
        page++;
    }
    
    return allRooms;
}

async function getGuilinElectricity(accessToken, roomId) {
    const serverUrl = getServerForRoom(roomId);
    console.log(`[桂林校区电费查询] roomId: ${roomId}, serverUrl: ${serverUrl}`);

    return retryRequest(async () => {
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };

        const formData = new FormData();
        formData.append('room_id', roomId);

        const response = await axios.post(`${serverUrl}/v1/cgElec/elec/query`, formData, {
            headers,
            timeout: 15000
        });

        console.log(`[桂林校区电费查询响应] ${JSON.stringify(response.data)}`);

        if (response.data) {
            return response.data.data || response.data || null;
        }
        return null;
    }, '桂林校区电费查询');
}

module.exports = {
    GUILIN_SERVER_1,
    GUILIN_SERVER_2,
    GUILIN_BUILDING_SERVER_MAP,
    getServerForBuilding,
    getServerForRoom,
    getGuilinBuildings,
    getGuilinRooms,
    getGuilinAllRoomsByBuilding,
    getGuilinElectricity
};
