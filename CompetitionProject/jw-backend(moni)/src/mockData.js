const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, 'data');

function readCSVFile(filename) {
    const filePath = path.join(DATA_DIR, filename);
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.trim().split('\n');
        const headers = lines[0].split(',');
        
        return lines.slice(1).map(line => {
            const values = line.split(',');
            const obj = {};
            headers.forEach((header, index) => {
                obj[header] = values[index] || '';
            });
            return obj;
        });
    } catch (error) {
        console.error(`读取CSV文件失败 ${filename}:`, error.message);
        return [];
    }
}

function readCSVWithSemester(filename, keyField = 'semester') {
    const data = readCSVFile(filename);
    const result = {};
    data.forEach(item => {
        const key = item[keyField];
        if (!result[key]) {
            result[key] = [];
        }
        const itemCopy = { ...item };
        delete itemCopy[keyField];
        result[key].push(itemCopy);
    });
    return result;
}

const mockStudents = readCSVFile('students.csv');

const mockTimetable = readCSVFile('timetable.csv');

const mockGrades = readCSVWithSemester('grades.csv');

const mockExams = readCSVFile('exams.csv');

const mockPlans = readCSVWithSemester('plans.csv');

const mockProgress = readCSVFile('progress.csv');

const mockDormitoryBuildings = readCSVFile('dormitory_buildings.csv');

const mockAnnouncements = readCSVFile('announcements.csv').map(item => ({
    id: parseInt(item.id),
    title: item.title,
    url: item.url,
    date: item.date
}));

const mockAnnouncementDetails = [
    {
        title: '关于2025-2026学年第一学期期末考试安排的通知',
        content: `<p>各学院、全体同学：</p><p>2025-2026学年第一学期期末考试将于2025年1月10日至1月20日进行，现将有关安排通知如下：</p><h3>一、考试时间</h3><p>2025年1月10日至1月20日，具体时间以课程表为准。</p><h3>二、考试地点</h3><p>各课程考试地点详见《考试安排表》。</p><h3>三、注意事项</h3><ol><li>考生须携带本人学生证和身份证参加考试。</li><li>考试开始15分钟后不得进入考场。</li><li>严禁携带手机等通讯工具进入考场。</li><li>遵守考场纪律，服从监考老师管理。</li></ol><p>请各学院及时通知学生，做好考试准备工作。</p><p>教务处</p><p>2025年6月1日</p>`,
        date: '2025-06-01',
        attachments: [
            { name: '考试安排表.xlsx', url: 'https://jwc.bwgl.cn/files/exam_schedule.xlsx' },
            { name: '考场规则.pdf', url: 'https://jwc.bwgl.cn/files/exam_rules.pdf' }
        ],
        url: 'https://jwc.bwgl.cn/announcement/20250601'
    },
    {
        title: '2025年暑期社会实践活动报名通知',
        content: `<p>各学院、全体同学：</p><p>为深入贯彻落实党的二十大精神，引导广大青年学生在实践中受教育、长才干、作贡献，学校决定组织开展2025年暑期社会实践活动。现将有关事项通知如下：</p><h3>一、活动主题</h3><p>青春心向党·建功新时代</p><h3>二、活动时间</h3><p>2025年7月10日至8月20日</p><h3>三、活动内容</h3><ol><li>乡村振兴实践</li><li>科技支农服务</li><li>教育关爱服务</li><li>文化宣传服务</li><li>疫情防控服务</li></ol><h3>四、报名方式</h3><p>请各学院组织学生于6月15日前通过学校实践平台完成报名。</p><p>联系人：张老师，联系电话：0771-12345678</p><p>校团委</p><p>2025年5月28日</p>`,
        date: '2025-05-28',
        attachments: [
            { name: '社会实践活动方案.docx', url: 'https://jwc.bwgl.cn/files/social_practice.docx' }
        ],
        url: 'https://jwc.bwgl.cn/announcement/20250528'
    }
];

const mockEmptyRooms = {
    '星期一': [
        { room: '教A101', periods: ['01-02', '05-06', '09-10'] },
        { room: '教A102', periods: ['03-04', '07-08', '11-12'] },
        { room: '教A103', periods: ['01-02', '03-04', '05-06'] },
        { room: '教B201', periods: ['07-08', '09-10', '11-12'] },
        { room: '教B202', periods: ['01-02', '09-10'] },
        { room: '教C301', periods: ['03-04', '05-06', '07-08'] }
    ],
    '星期二': [
        { room: '教A101', periods: ['03-04', '07-08'] },
        { room: '教A102', periods: ['01-02', '05-06', '11-12'] },
        { room: '教A103', periods: ['09-10', '11-12'] },
        { room: '教B201', periods: ['01-02', '03-04', '05-06'] },
        { room: '教B202', periods: ['07-08', '09-10'] },
        { room: '教C301', periods: ['01-02', '03-04'] }
    ],
    '星期三': [
        { room: '教A101', periods: ['05-06', '09-10', '11-12'] },
        { room: '教A102', periods: ['01-02', '03-04'] },
        { room: '教A103', periods: ['07-08', '09-10', '11-12'] },
        { room: '教B201', periods: ['01-02', '05-06'] },
        { room: '教B202', periods: ['03-04', '07-08'] },
        { room: '教C301', periods: ['09-10', '11-12'] }
    ],
    '星期四': [
        { room: '教A101', periods: ['01-02', '07-08', '11-12'] },
        { room: '教A102', periods: ['03-04', '05-06', '09-10'] },
        { room: '教A103', periods: ['01-02', '03-04'] },
        { room: '教B201', periods: ['05-06', '07-08', '09-10'] },
        { room: '教B202', periods: ['01-02', '11-12'] },
        { room: '教C301', periods: ['03-04', '05-06'] }
    ],
    '星期五': [
        { room: '教A101', periods: ['03-04', '09-10'] },
        { room: '教A102', periods: ['01-02', '05-06', '07-08', '11-12'] },
        { room: '教A103', periods: ['03-04', '07-08', '09-10'] },
        { room: '教B201', periods: ['01-02', '03-04', '11-12'] },
        { room: '教B202', periods: ['05-06', '09-10'] },
        { room: '教C301', periods: ['01-02', '07-08'] }
    ],
    '星期六': [
        { room: '教A101', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教A102', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教A103', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教B201', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教B202', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教C301', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] }
    ],
    '星期日': [
        { room: '教A101', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教A102', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教A103', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教B201', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教B202', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] },
        { room: '教C301', periods: ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'] }
    ]
};

function generateMockRooms(buildingId, buildingName) {
    const rooms = [];
    const floors = buildingId.startsWith('H') ? 6 : (buildingId === 'B19' ? 10 : 6);
    const roomsPerFloor = 20;
    
    for (let floor = 1; floor <= floors; floor++) {
        for (let room = 1; room <= roomsPerFloor; room++) {
            const roomNum = floor * 100 + room;
            const roomId = `H${buildingId}${roomNum}`;
            rooms.push({
                room_id: roomId,
                room_name: `${buildingName}${roomNum}`,
                floor: floor,
                capacity: 4,
                occupied: Math.random() > 0.3
            });
        }
    }
    return rooms;
}

function getMockElectricity(roomId) {
    return {
        room_id: roomId,
        balance: (Math.random() * 100 + 10).toFixed(2),
        last_update: new Date().toISOString(),
        status: '正常',
        unit: '元',
        warning_threshold: 20
    };
}

function getMockElectricityReminderSettings(studentId) {
    return {
        studentId: studentId,
        enabled: Math.random() > 0.5,
        threshold: Math.floor(Math.random() * 20) + 5,
        electricityAccount: `${studentId}@elec`,
        electricityPassword: '123456',
        roomId: `H4320${Math.floor(Math.random() * 600) + 101}`,
        campusId: 'nnxq',
        buildingId: '4320'
    };
}

function getMockUserPushToken(studentId) {
    return {
        studentId: studentId,
        pushToken: `push_${studentId}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        deviceInfo: 'HarmonyOS Device',
        isActive: true
    };
}

function getMockEmptyRooms(dayOfWeek) {
    return mockEmptyRooms[dayOfWeek] || [];
}

module.exports = {
    mockStudents,
    mockTimetable,
    mockGrades,
    mockExams,
    mockPlans,
    mockProgress,
    mockDormitoryBuildings,
    mockAnnouncements,
    mockAnnouncementDetails,
    mockEmptyRooms,
    generateMockRooms,
    getMockElectricity,
    getMockElectricityReminderSettings,
    getMockUserPushToken,
    getMockEmptyRooms
};