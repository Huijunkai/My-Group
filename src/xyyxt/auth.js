const { BASE_URL, AUTH_URL, ELEC_URL, ELEC_URL_NANNING, ELEC_URL_GUILIN, BASIC_AUTH, DEFAULT_HEADERS, CAMPUS_CONFIG, GUILIN_SERVER_1, GUILIN_SERVER_2, GUILIN_BUILDING_SERVER_MAP } = require('./constants');
const axios = require('axios');
const FormData = require('form-data');

function getElecUrl(areaId = '') {
    if (areaId && CAMPUS_CONFIG[areaId]) {
        return CAMPUS_CONFIG[areaId].elecUrl;
    }
    return ELEC_URL;
}

const NANNING_ELEC_URL_1 = 'http://202.103.236.36:10002';

const NANNING_ELEC_URL_2 = 'http://202.103.236.36:10001';

const NANNING_BUILDING_URL_MAP = {
    '4320': NANNING_ELEC_URL_1,
    '4523': NANNING_ELEC_URL_1,
    '4722': NANNING_ELEC_URL_1,
    '5158': NANNING_ELEC_URL_1,
    '5623': NANNING_ELEC_URL_1,
    '6068': NANNING_ELEC_URL_1,
    '6267': NANNING_ELEC_URL_1,
    '6454': NANNING_ELEC_URL_1,
    '6899': NANNING_ELEC_URL_1,
    'B1': NANNING_ELEC_URL_2,
    'B2': NANNING_ELEC_URL_2,
    'B3': NANNING_ELEC_URL_2,
    'B4': NANNING_ELEC_URL_2,
    'B5': NANNING_ELEC_URL_2,
    'B6': NANNING_ELEC_URL_2,
    'B7': NANNING_ELEC_URL_2,
    'B8': NANNING_ELEC_URL_2,
    'B9': NANNING_ELEC_URL_2,
    'B10': NANNING_ELEC_URL_2,
    'B11': NANNING_ELEC_URL_2,
    'B12': NANNING_ELEC_URL_2,
    'B16': NANNING_ELEC_URL_2,
    'B19': NANNING_ELEC_URL_2
};

function getElecUrlForRoom(areaId = '', roomId = '', buildingId = '') {
    if (areaId === 'nnxq') {
        if (buildingId && NANNING_BUILDING_URL_MAP[buildingId]) {
            return NANNING_BUILDING_URL_MAP[buildingId];
        }
        
        if (roomId) {
            for (const [bid, url] of Object.entries(NANNING_BUILDING_URL_MAP)) {
                if (roomId.startsWith('H' + bid)) {
                    return url;
                }
            }
            for (const [bid, url] of Object.entries(NANNING_BUILDING_URL_MAP)) {
                if (roomId.includes(bid)) {
                    return url;
                }
            }
        }
        
        console.warn(`[getElecUrlForRoom] 无法确定南宁校区楼栋服务器，buildingId: ${buildingId}, roomId: ${roomId}`);
        return NANNING_ELEC_URL_2;
    }
    
    if (areaId === 'glxq' && roomId) {
        for (const [buildingId, serverUrl] of Object.entries(GUILIN_BUILDING_SERVER_MAP)) {
            if (roomId.includes(buildingId) || roomId.startsWith('H' + buildingId)) {
                return serverUrl;
            }
        }
        if (roomId.startsWith('H') && roomId.length === 5) {
            return GUILIN_SERVER_1;
        }
        if (roomId.startsWith('H') && roomId.length === 4) {
            return GUILIN_SERVER_2;
        }
    }
    
    if (!areaId && roomId) {
        for (const [buildingId, serverUrl] of Object.entries(GUILIN_BUILDING_SERVER_MAP)) {
            if (roomId.includes(buildingId) || roomId.startsWith('H' + buildingId)) {
                return serverUrl;
            }
        }
        for (const [buildingId, url] of Object.entries(NANNING_BUILDING_URL_MAP)) {
            if (roomId.startsWith('H' + buildingId) || roomId.includes(buildingId)) {
                return url;
            }
        }
    }
    
    return getElecUrl(areaId);
}

function getRoomUrl(areaId = '', buildingId = '') {
    if (areaId === 'nnxq' && buildingId && NANNING_BUILDING_URL_MAP[buildingId]) {
        return NANNING_BUILDING_URL_MAP[buildingId];
    }
    if (areaId && CAMPUS_CONFIG[areaId] && CAMPUS_CONFIG[areaId].roomUrl) {
        return CAMPUS_CONFIG[areaId].roomUrl;
    }
    return getElecUrl(areaId);
}

function formatCookies(cookies) {
    if (!cookies) return '';
    if (Array.isArray(cookies)) {
        return cookies.map(c => c.split(';')[0]).join('; ');
    }
    return cookies;
}

function createInstance(cookies = '', referer = '', authHeader = '') {
    const headers = { ...DEFAULT_HEADERS };
    if (cookies) {
        headers['Cookie'] = formatCookies(cookies);
    }
    if (referer) {
        headers['Referer'] = referer;
    }
    if (authHeader) {
        headers['Authorization'] = authHeader;
    }

    return axios.create({
        headers,
        withCredentials: true,
        timeout: 30000
    });
}

async function login(username, password) {
    try {
        const instance = createInstance('', '', BASIC_AUTH);
        
        const loginUrl = `${AUTH_URL}/authentication/form`;
        
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        formData.append('grant_type', 'password');
        
        const response = await instance.post(loginUrl, formData.toString(), {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        if (response.data && response.data.access_token) {
            const cookies = response.headers['set-cookie'] || [];
            return {
                success: true,
                cookies: cookies,
                data: {
                    access_token: response.data.access_token,
                    refresh_token: response.data.refresh_token,
                    expires_in: response.data.expires_in,
                    schoolId: response.data.schoolId,
                    userId: response.data.userId
                }
            };
        } else {
            return {
                success: false,
                message: response.data?.message || response.data?.error_description || '登录失败，请检查账号密码'
            };
        }
    } catch (error) {
        console.error('校园一信通登录失败:', error.message);
        if (error.response) {
            console.error('响应状态:', error.response.status);
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return {
            success: false,
            message: error.response?.data?.message || error.response?.data?.error_description || error.message || '登录请求失败'
        };
    }
}

async function refreshToken(refreshToken) {
    try {
        const instance = createInstance('', '', BASIC_AUTH);
        
        const url = `${AUTH_URL}/authentication/refresh`;
        
        const formData = new URLSearchParams();
        formData.append('grant_type', 'refresh_token');
        formData.append('refresh_token', refreshToken);
        
        const response = await instance.post(url, formData.toString(), {
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
        });

        if (response.data && response.data.access_token) {
            return {
                success: true,
                data: response.data
            };
        } else {
            return {
                success: false,
                message: response.data?.message || '刷新Token失败'
            };
        }
    } catch (error) {
        console.error('刷新Token失败:', error.message);
        return {
            success: false,
            message: error.message || '刷新Token请求失败'
        };
    }
}

async function getUserInfo(accessToken) {
    try {
        const instance = createInstance('', '', `bearer ${accessToken}`);
        const response = await instance.get(`${AUTH_URL}/user/info`);
        
        if (response.data && response.data.retCode === 200) {
            return response.data.data || null;
        }
        return null;
    } catch (error) {
        console.error('获取用户信息失败:', error.message);
        return null;
    }
}

async function getBalance(accessToken) {
    try {
        const instance = createInstance('', '', `bearer ${accessToken}`);
        const response = await instance.get(`${BASE_URL}/bwgl_remoteservice/cardController/getAccountBln`);
        
        if (response.data && response.data.code === 200) {
            return response.data.data || response.data || null;
        }
        return response.data || null;
    } catch (error) {
        console.error('获取余额失败:', error.message);
        return null;
    }
}

async function getTransactions(accessToken, page = 1, size = 20) {
    try {
        const instance = createInstance('', '', `bearer ${accessToken}`);
        const response = await instance.get(`${BASE_URL}/yxtapp/mobile/card/transactions`, {
            params: { page, size }
        });
        
        if (response.data && response.data.retCode === 200) {
            return response.data.data || [];
        }
        return [];
    } catch (error) {
        console.error('获取交易记录失败:', error.message);
        return [];
    }
}

async function getConsumptionRecords(accessToken, page = 1, size = 20, timeRange = '') {
    try {
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        if (!timeRange) {
            const now = new Date();
            const endDate = now.toISOString().split('T')[0];
            const startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            timeRange = `${startDate}~${endDate}`;
        }
        
        const url = `${BASE_URL}/bwgl_remoteservice/consumption/newConsumption?opr=2&current=${page}&size=${size}&time=${encodeURIComponent(timeRange)}&reason=`;
        
        const formData = new FormData();
        
        const response = await axios.post(url, formData, { headers });
        
        console.log('消费记录响应:', JSON.stringify(response.data));
        
        if (response.data && response.data.code === 200) {
            return {
                data: response.data.data || [],
                total: response.data.total || 0,
                pages: response.data.pages || 0,
                current: response.data.current || page
            };
        }
        return { data: [], total: 0, pages: 0, current: page };
    } catch (error) {
        console.error('获取消费记录失败:', error.message);
        if (error.response) {
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getRechargeRecords(accessToken, page = 1, size = 20, timeRange = '') {
    try {
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        if (!timeRange) {
            const now = new Date();
            const endDate = now.toISOString().split('T')[0];
            const startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
            timeRange = `${startDate}~${endDate}`;
        }
        
        const url = `${BASE_URL}/bwgl_remoteservice/consumption/newConsumption?opr=1&current=${page}&size=${size}&time=${encodeURIComponent(timeRange)}&reason=`;
        
        const formData = new FormData();
        
        const response = await axios.post(url, formData, { headers });
        
        console.log('充值记录响应:', JSON.stringify(response.data));
        
        if (response.data && response.data.code === 200) {
            return {
                data: response.data.data || [],
                total: response.data.total || 0,
                pages: response.data.pages || 0,
                current: response.data.current || page
            };
        }
        return { data: [], total: 0, pages: 0, current: page };
    } catch (error) {
        console.error('获取充值记录失败:', error.message);
        if (error.response) {
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return { data: [], total: 0, pages: 0, current: page };
    }
}

const NANNING_BUILDINGS = [
    { loudong_id: '4320', loudong_name: '15-1栋' },
    { loudong_id: '4523', loudong_name: '15-2栋' },
    { loudong_id: '4722', loudong_name: '13-1栋' },
    { loudong_id: '5158', loudong_name: '13-2栋' },
    { loudong_id: '5623', loudong_name: '17栋' },
    { loudong_id: '6068', loudong_name: '18栋' },
    { loudong_id: '6267', loudong_name: '19栋' },
    { loudong_id: '6454', loudong_name: '20栋' },
    { loudong_id: '6899', loudong_name: '21栋' },
    { loudong_id: 'B1', loudong_name: '1号楼' },
    { loudong_id: 'B2', loudong_name: '2号楼' },
    { loudong_id: 'B3', loudong_name: '3号楼' },
    { loudong_id: 'B4', loudong_name: '4号楼' },
    { loudong_id: 'B5', loudong_name: '5号楼' },
    { loudong_id: 'B6', loudong_name: '6号楼' },
    { loudong_id: 'B7', loudong_name: '7号楼' },
    { loudong_id: 'B8', loudong_name: '8号楼' },
    { loudong_id: 'B9', loudong_name: '9号楼' },
    { loudong_id: 'B10', loudong_name: '10号楼' },
    { loudong_id: 'B11', loudong_name: '11号楼' },
    { loudong_id: 'B12', loudong_name: '12号楼' },
    { loudong_id: 'B16', loudong_name: '16号楼' },
    { loudong_id: 'B19', loudong_name: '14号楼' }
];

async function getBuildings(accessToken, areaId = '') {
    try {
        if (areaId === 'nnxq') {
            const roomUrl = getRoomUrl(areaId);
            const headers = {
                ...DEFAULT_HEADERS,
                'Authorization': `bearer ${accessToken}`,
                'Content-Type': 'multipart/form-data'
            };
            
            const formData = new FormData();
            formData.append('areaId', areaId);
            
            try {
                const response = await axios.post(`${roomUrl}/v1/cgElec/loudong/query`, formData, {
                    headers
                });
                
                const apiBuildings = response.data?.data || response.data || [];
                const existingIds = new Set(apiBuildings.map(b => b.loudong_id));
                const newBuildings = NANNING_BUILDINGS.filter(b => !existingIds.has(b.loudong_id));
                
                return [...apiBuildings, ...newBuildings];
            } catch (apiError) {
                console.error('南宁校区API获取楼栋失败:', apiError.message);
                return NANNING_BUILDINGS;
            }
        }
        
        const roomUrl = getRoomUrl(areaId);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        if (areaId) {
            formData.append('areaId', areaId);
        }
        
        const response = await axios.post(`${roomUrl}/v1/cgElec/loudong/query`, formData, {
            headers
        });
        
        if (response.data) {
            return response.data.data || response.data || [];
        }
        return [];
    } catch (error) {
        console.error('获取宿舍楼失败:', error.message);
        if (areaId === 'nnxq') {
            return NANNING_BUILDINGS;
        }
        return [];
    }
}

async function getRooms(accessToken, buildingId, areaId = '', page = 1, size = 100) {
    try {
        const roomUrl = getRoomUrl(areaId, buildingId);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        formData.append('loudong_id', buildingId);
        formData.append('current', page);
        formData.append('size', size);
        
        const response = await axios.post(`${roomUrl}/v1/cgElec/room/query`, formData, {
            headers
        });
        
        if (response.data) {
            return {
                data: response.data.data || [],
                total: response.data.total || 0,
                pages: response.data.pages || 0,
                current: response.data.current || page
            };
        }
        return { data: [], total: 0, pages: 0, current: page };
    } catch (error) {
        console.error('获取宿舍房间失败:', error.message);
        if (error.response) {
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getNanningRooms(accessToken, buildingId, page = 1, size = 100) {
    const roomUrl = getRoomUrl('nnxq', buildingId);
    const headers = {
        ...DEFAULT_HEADERS,
        'Authorization': `bearer ${accessToken}`,
        'Content-Type': 'multipart/form-data'
    };
    
    const tryParams = [
        { dormitoryBdId: buildingId },
        { loudong_id: buildingId },
        { buildingId: buildingId }
    ];
    
    for (const params of tryParams) {
        try {
            const formData = new FormData();
            for (const [key, value] of Object.entries(params)) {
                formData.append(key, value);
            }
            formData.append('current', page);
            formData.append('size', size);
            
            console.log(`尝试南宁房间查询 URL: ${roomUrl}/v1/cgElec/room/query, 参数: ${JSON.stringify(params)}`);
            
            const response = await axios.post(`${roomUrl}/v1/cgElec/room/query`, formData, {
                headers,
                timeout: 10000
            });
            
            console.log(`响应状态: ${response.status}, 数据: ${JSON.stringify(response.data).substring(0, 500)}`);
            
            if (response.data && response.data.data && response.data.data.length > 0) {
                console.log(`成功获取 ${response.data.data.length} 个房间`);
                return {
                    data: response.data.data,
                    total: response.data.total || response.data.data.length,
                    pages: response.data.pages || 1,
                    current: response.data.current || page
                };
            }
        } catch (error) {
            console.error(`参数 ${JSON.stringify(params)} 失败:`, error.message);
            if (error.response) {
                console.error(`响应状态: ${error.response.status}, 数据: ${JSON.stringify(error.response.data)}`);
            }
        }
    }
    
    console.log(`南宁校区 ${buildingId} API查询失败，使用硬编码数据`);
    return { data: [], total: 0, pages: 0, current: page };
}

async function getRoomsByDormitoryBdId(accessToken, dormitoryBdId, areaId = '') {
    try {
        const elecUrl = getElecUrl(areaId);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        formData.append('dormitoryBdId', dormitoryBdId);
        
        const response = await axios.post(`${elecUrl}/v1/cgElec/room/query`, formData, {
            headers
        });
        
        console.log(`getRoomsByDormitoryBdId response for ${dormitoryBdId}:`, JSON.stringify(response.data));
        
        if (response.data && response.data.data) {
            return response.data.data || [];
        }
        return [];
    } catch (error) {
        console.error('getRoomsByDormitoryBdId失败:', error.message);
        return [];
    }
}

function generateRooms(prefix, buildingName, floors, roomsPerFloor) {
    const rooms = [];
    for (let floor = 1; floor <= floors; floor++) {
        for (let room = 1; room <= roomsPerFloor; room++) {
            const roomNum = floor * 100 + room;
            const roomId = `${prefix}${roomNum}`;
            const roomName = `${buildingName}${roomNum}`;
            rooms.push({ room_id: roomId, room_name: roomName });
        }
    }
    return rooms;
}

const NANNING_BUILDING_ROOMS = {
    '4320': generateRooms('H', '15-1-', 6, 20),
    '4523': generateRooms('H', '15-2-', 6, 20),
    '4722': generateRooms('H', '13-1-', 6, 20),
    '5158': generateRooms('H', '13-2-', 6, 20),
    '5623': generateRooms('H', '17-', 6, 20),
    '6068': generateRooms('H', '18-', 6, 20),
    '6267': generateRooms('H', '19-', 6, 20),
    '6454': generateRooms('H', '20-', 6, 20),
    '6899': generateRooms('H', '21-', 6, 20),
    'B1': generateRooms('H', '1-', 6, 20),
    'B2': generateRooms('H', '2-', 6, 20),
    'B3': generateRooms('H', '3-', 6, 20),
    'B4': generateRooms('H', '4-', 6, 20),
    'B5': generateRooms('H', '5-', 6, 20),
    'B6': generateRooms('H', '6-', 6, 20),
    'B7': generateRooms('H', '7-', 6, 20),
    'B8': generateRooms('H', '8-', 6, 20),
    'B9': generateRooms('H', '9-', 6, 20),
    'B10': generateRooms('H', '10-', 6, 20),
    'B11': generateRooms('H', '11-', 6, 20),
    'B12': generateRooms('H', '12-', 6, 20),
    'B16': generateRooms('H', '16-', 6, 20),
    'B19': generateRooms('H14', '14-', 10, 20)
};

async function getAllRoomsByBuilding(accessToken, buildingId, areaId = '') {
    if (areaId === 'nnxq') {
        const allRooms = [];
        let page = 1;
        const size = 100;
        
        while (true) {
            const result = await getNanningRooms(accessToken, buildingId, page, size);
            if (result.data && result.data.length > 0) {
                allRooms.push(...result.data);
            }
            if (page >= result.pages || result.data.length === 0) {
                break;
            }
            page++;
        }
        
        if (allRooms.length > 0) {
            console.log(`南宁校区 ${buildingId} 从API获取到 ${allRooms.length} 个房间`);
            return allRooms;
        }
        
        if (NANNING_BUILDING_ROOMS[buildingId]) {
            console.log(`南宁校区 ${buildingId} 使用硬编码数据`);
            return NANNING_BUILDING_ROOMS[buildingId];
        }
        
        return [];
    }
    
    const allRooms = [];
    let page = 1;
    const size = 100;
    
    while (true) {
        const result = await getRooms(accessToken, buildingId, areaId, page, size);
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

const BUILDINGS = {
    '15-1栋': '4320',
    '15-2栋': '4523',
    '13-1栋': '4722',
    '13-2栋': '5158',
    '17栋': '5623',
    '18栋': '6068',
    '19栋': '6267',
    '20栋': '6454',
    '21栋': '6899'
};

async function getAllBuildingsRooms(accessToken, areaId = 'nnxq') {
    const result = {};
    
    for (const [name, id] of Object.entries(BUILDINGS)) {
        console.log(`获取 ${name} 房间列表...`);
        const rooms = await getAllRoomsByBuilding(accessToken, id, areaId);
        result[name] = {
            buildingId: id,
            rooms: rooms
        };
    }
    
    return result;
}

async function getElectricity(accessToken, roomId, areaId = '', buildingId = '') {
    try {
        const elecUrl = getElecUrlForRoom(areaId, roomId, buildingId);
        console.log(`[电费查询] roomId: ${roomId}, areaId: ${areaId}, buildingId: ${buildingId}, elecUrl: ${elecUrl}`);
        const headers = {
            ...DEFAULT_HEADERS,
            'Authorization': `bearer ${accessToken}`,
            'Content-Type': 'multipart/form-data'
        };
        
        const formData = new FormData();
        formData.append('room_id', roomId);
        
        const response = await axios.post(`${elecUrl}/v1/cgElec/elec/query`, formData, {
            headers
        });
        
        console.log(`[电费查询响应] ${JSON.stringify(response.data)}`);
        
        if (response.data) {
            return response.data.data || response.data || null;
        }
        return null;
    } catch (error) {
        console.error('获取电费余额失败:', error.message);
        if (error.response) {
            console.error('响应状态:', error.response.status);
            console.error('响应数据:', JSON.stringify(error.response.data));
        }
        return null;
    }
}

module.exports = {
    login,
    refreshToken,
    getUserInfo,
    getBalance,
    getTransactions,
    getConsumptionRecords,
    getRechargeRecords,
    getBuildings,
    getRooms,
    getAllRoomsByBuilding,
    getAllBuildingsRooms,
    BUILDINGS,
    getElectricity,
    createInstance,
    formatCookies
};
