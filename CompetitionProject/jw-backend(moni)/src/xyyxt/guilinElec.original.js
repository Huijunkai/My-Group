const { mockDormitoryBuildings, generateMockRooms, getMockElectricity } = require('../mockData');

async function getGuilinBuildings(accessToken) {
    try {
        console.log('[Mock Guilin] 获取桂林校区楼栋列表');
        
        await new Promise(resolve => setTimeout(resolve, 400));
        
        return mockDormitoryBuildings.filter(b => b.xiaoqu_id === 'glxq');
    } catch (error) {
        console.error('[Mock Guilin] 获取桂林校区楼栋失败:', error.message);
        return [];
    }
}

async function getGuilinRooms(accessToken, buildingId, page = 1, size = 100) {
    try {
        console.log(`[Mock Guilin] 获取桂林校区房间 - 楼栋: ${buildingId}, 页码: ${page}`);
        
        await new Promise(resolve => setTimeout(resolve, 450));
        
        const building = mockDormitoryBuildings.find(b => b.loudong_id === buildingId && b.xiaoqu_id === 'glxq');
        if (!building) {
            return { data: [], total: 0, pages: 0, current: page };
        }
        
        const allRooms = generateMockRooms(buildingId, building.loudong_name);
        const start = (page - 1) * size;
        const end = start + size;
        const paginatedRooms = allRooms.slice(start, end);
        
        return {
            data: paginatedRooms,
            total: allRooms.length,
            pages: Math.ceil(allRooms.length / size),
            current: page
        };
    } catch (error) {
        console.error('[Mock Guilin] 获取桂林校区房间失败:', error.message);
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getGuilinAllRoomsByBuilding(accessToken, buildingId) {
    try {
        console.log(`[Mock Guilin] 获取桂林校区楼栋所有房间 - 楼栋: ${buildingId}`);
        
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const building = mockDormitoryBuildings.find(b => b.loudong_id === buildingId && b.xiaoqu_id === 'glxq');
        if (!building) {
            return [];
        }
        
        return generateMockRooms(buildingId, building.loudong_name);
    } catch (error) {
        console.error('[Mock Guilin] 获取桂林校区楼栋所有房间失败:', error.message);
        return [];
    }
}

async function getGuilinElectricity(accessToken, roomId) {
    try {
        console.log(`[Mock Guilin] 查询桂林校区电费余额 - 房间: ${roomId}`);
        
        await new Promise(resolve => setTimeout(resolve, 350));
        
        return getMockElectricity(roomId);
    } catch (error) {
        console.error('[Mock Guilin] 查询桂林校区电费余额失败:', error.message);
        return null;
    }
}

module.exports = {
    GUILIN_SERVER_1: 'http://mock-server-1',
    GUILIN_SERVER_2: 'http://mock-server-2',
    GUILIN_BUILDING_SERVER_MAP: {},
    getServerForBuilding: () => 'http://mock-server',
    getServerForRoom: () => 'http://mock-server',
    getGuilinBuildings,
    getGuilinRooms,
    getGuilinAllRoomsByBuilding,
    getGuilinElectricity
};
