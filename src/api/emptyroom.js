const { BASE_URL } = require('../utils/constants');
const { createInstance } = require('../utils/request');
const cheerio = require('cheerio');

const CAMPUSES = [
    { code: '01', name: '桂林校区' },
    { code: 'oW', name: '南宁校区' }
];

const PERIOD_MAP = {
    '0102': [1, 2],
    '0304': [3, 4],
    '0506': [5, 6],
    '0708': [7, 8],
    '0910': [9, 10],
    '1112': [11, 12]
};

async function getCampuses() {
    return CAMPUSES;
}

async function getBuildings(cookies, campusCode) {
    if (!campusCode) {
        return [];
    }
    
    if (!cookies || cookies.length === 0) {
        return [];
    }
    
    try {
        const instance = createInstance(cookies, `${BASE_URL}/kbcx/kbxx_classroom`);
        const response = await instance.get(`${BASE_URL}/kbcx/getJxlByAjax?xqid=${campusCode}`);
        
        if (response.data && Array.isArray(response.data)) {
            return response.data.map(item => ({
                code: item.dm || '',
                name: item.dmmc || ''
            }));
        }
        return [];
    } catch (error) {
        console.error('获取教学楼失败:', error.message);
        return [];
    }
}

async function queryEmptyRooms(cookies, params) {
    if (!cookies || cookies.length === 0) {
        return [];
    }
    
    const { semester, campus, building, weekStart, weekEnd, periodStart, periodEnd } = params;
    
    try {
        const instance = createInstance(cookies, `${BASE_URL}/kbcx/kbxx_classroom`);
        
        const postData = new URLSearchParams();
        postData.append('xnxqh', semester || '');
        postData.append('skyx', '');
        postData.append('xqid', campus || '');
        postData.append('jzwid', building || '');
        postData.append('zc1', weekStart ? String(weekStart) : '');
        postData.append('zc2', weekEnd ? String(weekEnd) : '');
        postData.append('jc1', periodStart || '');
        postData.append('jc2', periodEnd || '');
        
        const response = await instance.post(`${BASE_URL}/kbcx/kbxx_classroom_ifr`, postData);
        
        return parseEmptyRooms(response.data);
    } catch (error) {
        console.error('查询空教室失败:', error.message);
        return [];
    }
}

async function queryRoomSchedule(cookies, params) {
    if (!cookies || cookies.length === 0) {
        return null;
    }
    
    const { roomName, semester, campus, building, weekStart, weekEnd, periodStart, periodEnd } = params;
    
    try {
        const instance = createInstance(cookies, `${BASE_URL}/kbcx/kbxx_classroom`);
        
        const postData = new URLSearchParams();
        postData.append('xnxqh', semester || '');
        postData.append('skyx', '');
        postData.append('xqid', campus || '');
        postData.append('jzwid', building || '');
        postData.append('zc1', weekStart ? String(weekStart) : '');
        postData.append('zc2', weekEnd ? String(weekEnd) : '');
        postData.append('jc1', periodStart || '');
        postData.append('jc2', periodEnd || '');
        
        const response = await instance.post(`${BASE_URL}/kbcx/kbxx_classroom_ifr`, postData);
        
        return parseRoomSchedule(response.data, roomName);
    } catch (error) {
        console.error('查询教室课表失败:', error.message);
        return null;
    }
}

function parseEmptyRooms(html) {
    const $ = cheerio.load(html);
    const emptyRooms = [];
    
    const table = $('#kbtable');
    if (table.length === 0) {
        return emptyRooms;
    }
    
    const rows = table.find('tr');
    
    const periodOrder = ['0102', '0304', '0506', '0708', '0910', '1112'];
    
    rows.each((rowIndex, row) => {
        if (rowIndex < 2) return;
        
        const cells = $(row).find('td');
        if (cells.length < 2) return;
        
        const roomCell = $(cells[0]);
        const roomNameFull = roomCell.text().trim();
        const roomName = extractRoomName(roomNameFull);
        
        if (!roomName) return;
        
        const emptySlots = [];
        
        for (let day = 0; day < 7; day++) {
            const dayEmptyPeriods = [];
            
            for (let periodIdx = 0; periodIdx < 6; periodIdx++) {
                const cellIndex = 1 + day * 6 + periodIdx;
                
                if (cellIndex >= cells.length) continue;
                
                const cell = $(cells[cellIndex]);
                
                if (isEmptyCell(cell)) {
                    const periodKey = periodOrder[periodIdx];
                    const periods = PERIOD_MAP[periodKey];
                    if (periods) {
                        dayEmptyPeriods.push(...periods);
                    }
                }
            }
            
            if (dayEmptyPeriods.length > 0) {
                emptySlots.push({
                    day: day + 1,
                    periods: [...new Set(dayEmptyPeriods)].sort((a, b) => a - b)
                });
            }
        }
        
        if (emptySlots.length > 0) {
            emptyRooms.push({
                roomName: roomName,
                building: '',
                campus: '',
                capacity: '',
                type: '',
                emptySlots: emptySlots
            });
        }
    });
    
    return emptyRooms;
}

function parseRoomSchedule(html, targetRoomName) {
    const $ = cheerio.load(html);
    
    const table = $('#kbtable');
    if (table.length === 0) {
        return null;
    }
    
    const rows = table.find('tr');
    const periodOrder = ['0102', '0304', '0506', '0708', '0910', '1112'];
    
    let foundRoom = null;
    
    rows.each((rowIndex, row) => {
        if (rowIndex < 2) return;
        if (foundRoom) return;
        
        const cells = $(row).find('td');
        if (cells.length < 2) return;
        
        const roomCell = $(cells[0]);
        const roomNameFull = roomCell.text().trim();
        const roomName = extractRoomName(roomNameFull);
        
        if (!roomName) return;
        
        if (roomName.toLowerCase() === targetRoomName.toLowerCase() || 
            roomName === targetRoomName) {
            const schedule = [];
            
            for (let day = 0; day < 7; day++) {
                const dayPeriods = [];
                
                for (let periodIdx = 0; periodIdx < 6; periodIdx++) {
                    const cellIndex = 1 + day * 6 + periodIdx;
                    
                    if (cellIndex >= cells.length) continue;
                    
                    const cell = $(cells[cellIndex]);
                    
                    if (!isEmptyCell(cell)) {
                        dayPeriods.push(periodOrder[periodIdx]);
                    }
                }
                
                if (dayPeriods.length > 0) {
                    schedule.push({
                        day: day + 1,
                        periods: dayPeriods
                    });
                }
            }
            
            foundRoom = {
                roomName: roomName,
                schedule: schedule
            };
        }
    });
    
    return foundRoom;
}

function extractRoomName(fullName) {
    if (!fullName) return null;
    
    const aMatch = fullName.match(/([A-Za-z]\d+)/);
    if (aMatch) {
        return aMatch[1];
    }
    
    const pureNumMatch = fullName.match(/(^|\D)(\d{2,})($|\D)/);
    if (pureNumMatch) {
        const num = pureNumMatch[2];
        if (/^\d+$/.test(num)) {
            return num;
        }
    }
    
    return null;
}

function isEmptyCell(cell) {
    const kbcontent = cell.find('.kbcontent1');
    
    if (kbcontent.length === 0) {
        return true;
    }
    
    const content = kbcontent.text().trim();
    if (!content || content === '' || content === '&nbsp;') {
        return true;
    }
    
    return false;
}

module.exports = {
    getCampuses,
    getBuildings,
    queryEmptyRooms,
    queryRoomSchedule
};
