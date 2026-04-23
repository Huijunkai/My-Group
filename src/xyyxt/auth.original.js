const { mockStudents, mockDormitoryBuildings, generateMockRooms, getMockElectricity } = require('../mockData');

async function login(username, password) {
    try {
        console.log(`[Mock XYYXT] 校园一信通登录: ${username}`);
        
        await new Promise(resolve => setTimeout(resolve, 600));
        
        const student = mockStudents.find(s => s.studentId === username && s.password === password);
        
        if (student) {
            console.log(`[Mock XYYXT] 登录成功: ${student.name}`);
            
            return {
                success: true,
                data: {
                    access_token: `MOCK_ACCESS_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    refresh_token: `MOCK_REFRESH_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    expires_in: 7200,
                    schoolId: '10001',
                    userId: student.studentId
                }
            };
        } else {
            console.log(`[Mock XYYXT] 登录失败: ${username}`);
            return {
                success: false,
                message: '账号或密码错误'
            };
        }
    } catch (error) {
        console.error('[Mock XYYXT] 登录异常:', error.message);
        return { success: false, message: error.message };
    }
}

async function getUserInfo(accessToken) {
    try {
        console.log('[Mock XYYXT] 获取用户信息');
        
        await new Promise(resolve => setTimeout(resolve, 300));
        
        const student = mockStudents[0];
        
        return {
            userId: student.studentId,
            name: student.name,
            gender: student.gender,
            college: student.college,
            major: student.major,
            className: student.className,
            enrollmentYear: student.enrollmentYear
        };
    } catch (error) {
        console.error('[Mock XYYXT] 获取用户信息失败:', error.message);
        return null;
    }
}

async function getBalance(accessToken) {
    try {
        console.log('[Mock XYYXT] 获取余额');
        
        await new Promise(resolve => setTimeout(resolve, 250));
        
        return {
            balance: (Math.random() * 500 + 50).toFixed(2),
            frozen: (Math.random() * 20).toFixed(2),
            available: (Math.random() * 480 + 30).toFixed(2),
            unit: '元',
            lastUpdate: new Date().toISOString()
        };
    } catch (error) {
        console.error('[Mock XYYXT] 获取余额失败:', error.message);
        return null;
    }
}

async function getTransactions(accessToken, page = 1, size = 20) {
    try {
        console.log(`[Mock XYYXT] 获取交易记录 - 页码: ${page}`);
        
        await new Promise(resolve => setTimeout(resolve, 350));
        
        const transactions = [];
        for (let i = 0; i < Math.min(size, 15); i++) {
            transactions.push({
                id: `TXN${Date.now()}${i}`,
                type: ['消费', '充值', '转账'][Math.floor(Math.random() * 3)],
                amount: (Math.random() * 50 - 10).toFixed(2),
                balance: (Math.random() * 500).toFixed(2),
                time: new Date(Date.now() - i * 86400000).toISOString(),
                description: ['食堂消费', '超市购物', '网费充值', '水费缴纳'][Math.floor(Math.random() * 4)]
            });
        }
        
        return transactions;
    } catch (error) {
        console.error('[Mock XYYXT] 获取交易记录失败:', error.message);
        return [];
    }
}

async function getConsumptionRecords(accessToken, page = 1, size = 20) {
    try {
        console.log(`[Mock XYYXT] 获取消费记录 - 页码: ${page}`);
        
        await new Promise(resolve => setTimeout(resolve, 320));
        
        const records = [];
        for (let i = 0; i < Math.min(size, 15); i++) {
            records.push({
                id: `CONSUME${Date.now()}${i}`,
                amount: (Math.random() * 30).toFixed(2),
                merchant: ['第一食堂', '第二食堂', '校园超市', '图书馆咖啡厅', '水果店'][Math.floor(Math.random() * 5)],
                time: new Date(Date.now() - i * 86400000).toISOString(),
                category: ['餐饮', '购物', '其他'][Math.floor(Math.random() * 3)]
            });
        }
        
        return {
            data: records,
            total: 150,
            pages: Math.ceil(150 / size),
            current: page
        };
    } catch (error) {
        console.error('[Mock XYYXT] 获取消费记录失败:', error.message);
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getRechargeRecords(accessToken, page = 1, size = 20) {
    try {
        console.log(`[Mock XYYXT] 获取充值记录 - 页码: ${page}`);
        
        await new Promise(resolve => setTimeout(resolve, 310));
        
        const records = [];
        for (let i = 0; i < Math.min(size, 10); i++) {
            records.push({
                id: `RECHARGE${Date.now()}${i}`,
                amount: [50, 100, 200, 500][Math.floor(Math.random() * 4)],
                method: ['银行卡', '支付宝', '微信支付'][Math.floor(Math.random() * 3)],
                time: new Date(Date.now() - i * 7 * 86400000).toISOString(),
                status: '成功'
            });
        }
        
        return {
            data: records,
            total: 45,
            pages: Math.ceil(45 / size),
            current: page
        };
    } catch (error) {
        console.error('[Mock XYYXT] 获取充值记录失败:', error.message);
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getBuildings(accessToken, areaId = '') {
    try {
        console.log(`[Mock XYYXT] 获取宿舍楼列表 - 校区: ${areaId || '全部'}`);
        
        await new Promise(resolve => setTimeout(resolve, 400));
        
        if (areaId) {
            return mockDormitoryBuildings.filter(b => b.xiaoqu_id === areaId);
        }
        
        return mockDormitoryBuildings;
    } catch (error) {
        console.error('[Mock XYYXT] 获取宿舍楼失败:', error.message);
        return [];
    }
}

async function getRooms(accessToken, buildingId, areaId = '', page = 1, size = 100) {
    try {
        console.log(`[Mock XYYXT] 获取房间列表 - 楼栋: ${buildingId}, 页码: ${page}`);
        
        await new Promise(resolve => setTimeout(resolve, 450));
        
        const building = mockDormitoryBuildings.find(b => b.loudong_id === buildingId);
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
        console.error('[Mock XYYXT] 获取房间失败:', error.message);
        return { data: [], total: 0, pages: 0, current: page };
    }
}

async function getAllRoomsByBuilding(accessToken, buildingId, areaId = '') {
    try {
        console.log(`[Mock XYYXT] 获取楼栋所有房间 - 楼栋: ${buildingId}`);
        
        await new Promise(resolve => setTimeout(resolve, 500));
        
        const building = mockDormitoryBuildings.find(b => b.loudong_id === buildingId);
        if (!building) {
            return [];
        }
        
        return generateMockRooms(buildingId, building.loudong_name);
    } catch (error) {
        console.error('[Mock XYYXT] 获取楼栋所有房间失败:', error.message);
        return [];
    }
}

async function getAllBuildingsRooms(accessToken, areaId = 'nnxq') {
    try {
        console.log(`[Mock XYYXT] 获取校区所有楼栋房间 - 校区: ${areaId}`);
        
        await new Promise(resolve => setTimeout(resolve, 800));
        
        const result = {};
        const buildings = areaId ? 
            mockDormitoryBuildings.filter(b => b.xiaoqu_id === areaId) : 
            mockDormitoryBuildings;
        
        for (const building of buildings) {
            result[building.loudong_name] = {
                buildingId: building.loudong_id,
                rooms: generateMockRooms(building.loudong_id, building.loudong_name)
            };
        }
        
        return result;
    } catch (error) {
        console.error('[Mock XYYXT] 获取校区所有楼栋房间失败:', error.message);
        return {};
    }
}

async function getElectricity(accessToken, roomId, areaId = '', buildingId = '') {
    try {
        console.log(`[Mock XYYXT] 查询电费余额 - 房间: ${roomId}`);
        
        await new Promise(resolve => setTimeout(resolve, 350));
        
        return getMockElectricity(roomId);
    } catch (error) {
        console.error('[Mock XYYXT] 查询电费余额失败:', error.message);
        return null;
    }
}

module.exports = {
    login,
    refreshToken: async () => ({ success: true, data: { access_token: 'MOCK_REFRESHED' } }),
    getUserInfo,
    getBalance,
    getTransactions,
    getConsumptionRecords,
    getRechargeRecords,
    getBuildings,
    getRooms,
    getAllRoomsByBuilding,
    getAllBuildingsRooms,
    BUILDINGS: {
        '15-1栋': '4320',
        '15-2栋': '4523',
        '13-1栋': '4722',
        '13-2栋': '5158',
        '17栋': '5623',
        '18栋': '6068',
        '19栋': '6267',
        '20栋': '6454',
        '21栋': '6899'
    },
    getElectricity,
    createInstance: () => {},
    formatCookies: () => ''
};
