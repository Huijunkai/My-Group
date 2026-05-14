const { getMockEmptyRooms } = require('../mockData');

const CAMPUSES = [
    { code: '01', name: '桂林校区' },
    { code: 'oW', name: '南宁校区' }
];

const BUILDINGS = {
    '01': [
        { code: '1', name: '第一教学楼' },
        { code: '2', name: '第二教学楼' },
        { code: '3', name: '第三教学楼' }
    ],
    'oW': [
        { code: 'A', name: '教A' },
        { code: 'B', name: '教B' },
        { code: 'C', name: '教C' },
        { code: 'D', name: '教D' }
    ]
};

async function getCampuses() {
    console.log('[Mock Empty Room] 获取校区列表');
    return CAMPUSES;
}

async function getBuildings(cookies, campusCode) {
    console.log('[Mock Empty Room] 获取教学楼列表 - 校区:', campusCode);
    
    if (!campusCode) {
        return [];
    }
    
    return BUILDINGS[campusCode] || [];
}

async function queryEmptyRooms(cookies, params) {
    console.log('[Mock Empty Room] 查询空教室 - 参数:', params);
    
    await new Promise(resolve => setTimeout(resolve, 300));
    
    const { weekStart, weekEnd, periodStart, periodEnd, campus, building } = params;
    
    // 模拟数据：按星期返回空教室
    const days = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];
    const result = [];
    
    days.forEach((day, index) => {
        const emptyRooms = getMockEmptyRooms(day);
        emptyRooms.forEach(roomInfo => {
            const periods = roomInfo.periods.filter(period => {
                // 过滤节次范围
                if (periodStart && periodEnd) {
                    const [start, end] = period.split('-').map(Number);
                    return start >= periodStart && end <= periodEnd;
                }
                return true;
            });
            
            if (periods.length > 0) {
                result.push({
                    roomName: roomInfo.room,
                    building: building || '教A',
                    campus: campus || 'oW',
                    capacity: 40,
                    type: '普通教室',
                    emptySlots: [{
                        day: index + 1,
                        periods: periods.map(p => {
                            const [start, end] = p.split('-').map(Number);
                            return Array.from({ length: end - start + 1 }, (_, i) => start + i);
                        }).flat()
                    }]
                });
            }
        });
    });
    
    return result;
}

async function queryRoomSchedule(cookies, params) {
    console.log('[Mock Empty Room] 查询教室课表 - 参数:', params);
    
    await new Promise(resolve => setTimeout(resolve, 200));
    
    const { roomName } = params;
    
    if (!roomName) {
        return null;
    }
    
    // 模拟数据：返回教室课表
    const schedule = [];
    const days = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];
    
    days.forEach((day, index) => {
        const emptyRooms = getMockEmptyRooms(day);
        const room = emptyRooms.find(r => r.room === roomName);
        
        if (room) {
            // 计算非空节次
            const allPeriods = ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'];
            const busyPeriods = allPeriods.filter(period => !room.periods.includes(period));
            
            if (busyPeriods.length > 0) {
                schedule.push({
                    day: index + 1,
                    periods: busyPeriods
                });
            }
        }
    });
    
    return {
        roomName: roomName,
        schedule: schedule
    };
}

module.exports = {
    getCampuses,
    getBuildings,
    queryEmptyRooms,
    queryRoomSchedule
};