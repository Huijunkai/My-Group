const mockStudents = [
    {
        studentId: '202101001',
        password: '123456',
        name: '张三',
        gender: '男',
        enrollmentYear: '2021',
        className: '计算机2101班',
        major: '计算机科学与技术',
        college: '信息工程学院'
    },
    {
        studentId: '202101002',
        password: '123456',
        name: '李四',
        gender: '女',
        enrollmentYear: '2021',
        className: '计算机2101班',
        major: '计算机科学与技术',
        college: '信息工程学院'
    },
    {
        studentId: '202102001',
        password: '123456',
        name: '王五',
        gender: '男',
        enrollmentYear: '2021',
        className: '软件工程2101班',
        major: '软件工程',
        college: '信息工程学院'
    },
    {
        studentId: '202103001',
        password: '123456',
        name: '赵六',
        gender: '女',
        enrollmentYear: '2021',
        className: '人工智能2101班',
        major: '人工智能',
        college: '信息工程学院'
    },
    {
        studentId: '202201001',
        password: '123456',
        name: '钱七',
        gender: '男',
        enrollmentYear: '2022',
        className: '计算机2201班',
        major: '计算机科学与技术',
        college: '信息工程学院'
    }
];

const mockTimetable = [
    {
        semester: '2025-1',
        name: '高等数学(下)',
        dayOfWeek: '星期一',
        week: 1,
        period: '01-02',
        teacher: '李教授',
        weeks: '1-16周',
        location: '教A201',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '数据结构',
        dayOfWeek: '星期一',
        week: 3,
        period: '03-04',
        teacher: '王教授',
        weeks: '1-18周',
        location: '教B305',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '大学英语(四)',
        dayOfWeek: '星期二',
        week: 1,
        period: '01-02',
        teacher: '张教授',
        weeks: '1-16周',
        location: '外语楼401',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '操作系统',
        dayOfWeek: '星期二',
        week: 3,
        period: '03-04',
        teacher: '刘教授',
        weeks: '1-18周',
        location: '教A301',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '计算机网络',
        dayOfWeek: '星期三',
        week: 1,
        period: '01-02',
        teacher: '陈教授',
        weeks: '1-16周',
        location: '教B201',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '数据库原理',
        dayOfWeek: '星期三',
        week: 3,
        period: '05-06',
        teacher: '赵教授',
        weeks: '1-18周',
        location: '机房C303',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '软件工程',
        dayOfWeek: '星期四',
        week: 1,
        period: '03-04',
        teacher: '孙教授',
        weeks: '1-16周',
        location: '教A401',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '算法设计与分析',
        dayOfWeek: '星期五',
        week: 1,
        period: '01-02',
        teacher: '周教授',
        weeks: '1-16周',
        location: '教B305',
        courseType: '选修课'
    },
    {
        semester: '2025-1',
        name: '编译原理',
        dayOfWeek: '星期五',
        week: 3,
        period: '07-08',
        teacher: '吴教授',
        weeks: '1-18周',
        location: '教A301',
        courseType: '必修课'
    },
    {
        semester: '2025-1',
        name: '人工智能导论',
        dayOfWeek: '星期六',
        week: 1,
        period: '09-10',
        teacher: '郑教授',
        weeks: '1-16周',
        location: '教B201',
        courseType: '选修课'
    },
    {
        semester: '2025-1',
        name: '机器学习',
        dayOfWeek: '星期日',
        week: 1,
        period: '11-12',
        teacher: '马教授',
        weeks: '1-16周',
        location: '机房C303',
        courseType: '选修课'
    },
    {
        semester: '2024-2',
        name: '高等数学(上)',
        dayOfWeek: '星期一',
        week: 1,
        period: '01-02',
        teacher: '李教授',
        weeks: '1-16周',
        location: '教A201',
        courseType: '必修课'
    },
    {
        semester: '2024-2',
        name: '线性代数',
        dayOfWeek: '星期二',
        week: 1,
        period: '03-04',
        teacher: '王教授',
        weeks: '1-16周',
        location: '教B201',
        courseType: '必修课'
    },
    {
        semester: '2024-1',
        name: '计算机导论',
        dayOfWeek: '星期三',
        week: 1,
        period: '01-02',
        teacher: '陈教授',
        weeks: '1-16周',
        location: '教A301',
        courseType: '必修课'
    },
    {
        semester: '2024-1',
        name: '英语(一)',
        dayOfWeek: '星期四',
        week: 1,
        period: '03-04',
        teacher: '张教授',
        weeks: '1-16周',
        location: '外语楼301',
        courseType: '必修课'
    },
    {
        semester: '2024-1',
        name: '体育(一)',
        dayOfWeek: '星期五',
        week: 1,
        period: '05-06',
        teacher: '刘教授',
        weeks: '1-16周',
        location: '体育馆',
        courseType: '必修课'
    }
];

const mockGrades = {
    '2024-2': [
        { courseCode: 'CS201', courseName: '高等数学(上)', score: '88', credit: '4', gradePoint: '4', examType: '3.7', courseType: '64' },
        { courseCode: 'CS202', courseName: '线性代数', score: '92', credit: '3', gradePoint: '3', examType: '3.9', courseType: '48' },
        { courseCode: 'CS203', courseName: '程序设计基础', score: '95', credit: '4', gradePoint: '4', examType: '4.0', courseType: '64' },
        { courseCode: 'CS204', courseName: '离散数学', score: '85', credit: '3', gradePoint: '3', examType: '3.5', courseType: '48' },
        { courseCode: 'CS205', courseName: '大学物理', score: '82', credit: '3', gradePoint: '3', examType: '3.3', courseType: '48' }
    ],
    '2024-1': [
        { courseCode: 'CS101', courseName: '计算机导论', score: '90', credit: '2', gradePoint: '2', examType: '3.8', courseType: '32' },
        { courseCode: 'CS102', courseName: '英语(一)', score: '87', credit: '3', gradePoint: '3', examType: '3.6', courseType: '48' },
        { courseCode: 'CS103', courseName: '体育(一)', score: '85', credit: '1', gradePoint: '1', examType: '3.5', courseType: '32' },
        { courseCode: 'CS104', courseName: '思想道德修养', score: '92', credit: '2', gradePoint: '2', examType: '3.9', courseType: '32' },
        { courseCode: 'CS105', courseName: '中国近代史', score: '88', credit: '2', gradePoint: '2', examType: '3.7', courseType: '32' }
    ],
    '2023-2': [
        { courseCode: 'CS001', courseName: '军事理论', score: '95', credit: '2', gradePoint: '2', examType: '4.0', courseType: '32' },
        { courseCode: 'CS002', courseName: '计算机基础', score: '92', credit: '3', gradePoint: '3', examType: '3.9', courseType: '48' }
    ]
};

const mockExams = [
    {
        courseName: '数据结构',
        examTime: '2025-01-10 09:00-11:00',
        location: '教A201',
        seatNumber: 'A15',
        examType: '闭卷考试',
        status: '未考试'
    },
    {
        courseName: '操作系统',
        examTime: '2025-01-12 14:00-16:00',
        location: '教B305',
        seatNumber: 'B22',
        examType: '闭卷考试',
        status: '未考试'
    },
    {
        courseName: '计算机网络',
        examTime: '2025-01-14 09:00-11:00',
        location: '教A301',
        seatNumber: 'A08',
        examType: '闭卷考试',
        status: '未考试'
    },
    {
        courseName: '数据库原理',
        examTime: '2025-01-16 14:00-16:30',
        location: '机房C303',
        seatNumber: 'C05',
        examType: '上机考试',
        status: '未考试'
    },
    {
        courseName: '高等数学(下)',
        examTime: '2025-01-18 09:00-11:30',
        location: '教A101',
        seatNumber: 'A30',
        examType: '闭卷考试',
        status: '未考试'
    },
    {
        courseName: '大学英语(四)',
        examTime: '2025-01-20 14:00-16:00',
        location: '外语楼501',
        seatNumber: 'E12',
        examType: '闭卷考试',
        status: '未考试'
    }
];

const mockPlans = {
    '2025-1': [
        { courseCode: 'CS301', courseName: '编译原理', teachingUnit: '信息工程学院', credit: '4', totalHours: '64', courseType: '必修课', courseAttribute: '专业核心课', examType: '闭卷考试', isExam: '是' },
        { courseCode: 'CS302', courseName: '人工智能导论', teachingUnit: '信息工程学院', credit: '3', totalHours: '48', courseType: '必修课', courseAttribute: '专业核心课', examType: '闭卷考试', isExam: '是' },
        { courseCode: 'CS303', courseName: '机器学习', teachingUnit: '信息工程学院', credit: '3', totalHours: '48', courseType: '选修课', courseAttribute: '专业选修课', examType: '课程设计', isExam: '是' },
        { courseCode: 'CS304', courseName: '计算机组成原理', teachingUnit: '信息工程学院', credit: '4', totalHours: '64', courseType: '必修课', courseAttribute: '专业核心课', examType: '闭卷考试', isExam: '是' },
        { courseCode: 'CS305', courseName: '嵌入式系统', teachingUnit: '信息工程学院', credit: '3', totalHours: '48', courseType: '选修课', courseAttribute: '专业选修课', examType: '课程设计', isExam: '是' }
    ],
    '2025-2': [
        { courseCode: 'CS401', courseName: '分布式系统', teachingUnit: '信息工程学院', credit: '3', totalHours: '48', courseType: '选修课', courseAttribute: '专业选修课', examType: '课程项目', isExam: '否' },
        { courseCode: 'CS402', courseName: '云计算技术', teachingUnit: '信息工程学院', credit: '2', totalHours: '32', courseType: '选修课', courseAttribute: '专业选修课', examType: '考查', isExam: '否' },
        { courseCode: 'CS403', courseName: '网络安全', teachingUnit: '信息工程学院', credit: '3', totalHours: '48', courseType: '必修课', courseAttribute: '专业核心课', examType: '闭卷考试', isExam: '是' },
        { courseCode: 'CS404', courseName: '软件工程实践', teachingUnit: '信息工程学院', credit: '4', totalHours: '64', courseType: '必修课', courseAttribute: '专业核心课', examType: '课程设计', isExam: '是' },
        { courseCode: 'CS405', courseName: '大数据分析', teachingUnit: '信息工程学院', credit: '3', totalHours: '48', courseType: '选修课', courseAttribute: '专业选修课', examType: '考查', isExam: '否' }
    ],
    '2026-1': [
        { courseCode: 'CS501', courseName: '毕业设计', teachingUnit: '信息工程学院', credit: '8', totalHours: '128', courseType: '必修课', courseAttribute: '专业核心课', examType: '答辩', isExam: '是' },
        { courseCode: 'CS502', courseName: '专业实习', teachingUnit: '信息工程学院', credit: '6', totalHours: '192', courseType: '必修课', courseAttribute: '实践教学', examType: '考查', isExam: '否' }
    ]
};

const mockProgress = [
    { category: '通识教育课程', requiredCredits: '40', completedCredits: '28', currentCredits: '8', remainingCredits: '12' },
    { category: '学科基础课程', requiredCredits: '50', completedCredits: '35', currentCredits: '9', remainingCredits: '15' },
    { category: '专业核心课程', requiredCredits: '45', completedCredits: '24', currentCredits: '12', remainingCredits: '21' },
    { category: '实践教学环节', requiredCredits: '25', completedCredits: '10', currentCredits: '6', remainingCredits: '15' },
    { category: '创新创业教育', requiredCredits: '8', completedCredits: '4', currentCredits: '2', remainingCredits: '4' },
    { category: '素质拓展课程', requiredCredits: '6', completedCredits: '2', currentCredits: '2', remainingCredits: '4' },
    { category: '专业选修课', requiredCredits: '16', completedCredits: '8', currentCredits: '4', remainingCredits: '8' }
];

const mockDormitoryBuildings = [
    { loudong_id: '4320', loudong_name: '15-1栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '4523', loudong_name: '15-2栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '4722', loudong_name: '13-1栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '5158', loudong_name: '13-2栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '5623', loudong_name: '17栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '6068', loudong_name: '18栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '6267', loudong_name: '19栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '6454', loudong_name: '20栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '6899', loudong_name: '21栋', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: 'B1', loudong_name: '1号楼', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: 'B2', loudong_name: '2号楼', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: 'B3', loudong_name: '3号楼', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: 'B4', loudong_name: '4号楼', xiaoqu_id: 'nnxq', xiaoqu_name: '南宁校区' },
    { loudong_id: '4320', loudong_name: '桂林校区9栋', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: '4509', loudong_name: '桂林校区7栋', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: '4722', loudong_name: '桂林校区12栋', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: '4812', loudong_name: '桂林校区13栋', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: '6436', loudong_name: '桂林校区14A栋', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: '6819', loudong_name: '桂林校区14B栋', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: 'B101', loudong_name: '桂林校区10A号楼', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: 'B102', loudong_name: '桂林校区10B号楼', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' },
    { loudong_id: 'B8', loudong_name: '桂林校区8号楼', xiaoqu_id: 'glxq', xiaoqu_name: '桂林校区' }
];

const mockAnnouncements = [
    {
        id: 1,
        title: '关于2025-2026学年第一学期期末考试安排的通知',
        url: 'https://jwc.bwgl.cn/announcement/20250601',
        date: '2025-06-01'
    },
    {
        id: 2,
        title: '2025年暑期社会实践活动报名通知',
        url: 'https://jwc.bwgl.cn/announcement/20250528',
        date: '2025-05-28'
    },
    {
        id: 3,
        title: '关于开展2025年度优秀学生奖学金评选工作的通知',
        url: 'https://jwc.bwgl.cn/announcement/20250520',
        date: '2025-05-20'
    },
    {
        id: 4,
        title: '2025届毕业生学位授予仪式安排',
        url: 'https://jwc.bwgl.cn/announcement/20250515',
        date: '2025-05-15'
    },
    {
        id: 5,
        title: '关于2025-2026学年第一学期选课的通知',
        url: 'https://jwc.bwgl.cn/announcement/20250510',
        date: '2025-05-10'
    },
    {
        id: 6,
        title: '2025年春季学期校历调整通知',
        url: 'https://jwc.bwgl.cn/announcement/20250425',
        date: '2025-04-25'
    },
    {
        id: 7,
        title: '关于组织参加2025年全国大学生计算机设计大赛的通知',
        url: 'https://jwc.bwgl.cn/announcement/20250420',
        date: '2025-04-20'
    },
    {
        id: 8,
        title: '2025年劳动节放假安排',
        url: 'https://jwc.bwgl.cn/announcement/20250415',
        date: '2025-04-15'
    },
    {
        id: 9,
        title: '关于开展2025年教师教学能力提升培训的通知',
        url: 'https://jwc.bwgl.cn/announcement/20250410',
        date: '2025-04-10'
    },
    {
        id: 10,
        title: '2025年春季学期中期教学检查安排',
        url: 'https://jwc.bwgl.cn/announcement/20250405',
        date: '2025-04-05'
    }
];

const mockAnnouncementDetails = [
    {
        title: '关于2025-2026学年第一学期期末考试安排的通知',
        content: `
            <p>各学院、全体同学：</p>
            <p>2025-2026学年第一学期期末考试将于2025年1月10日至1月20日进行，现将有关安排通知如下：</p>
            <h3>一、考试时间</h3>
            <p>2025年1月10日至1月20日，具体时间以课程表为准。</p>
            <h3>二、考试地点</h3>
            <p>各课程考试地点详见《考试安排表》。</p>
            <h3>三、注意事项</h3>
            <ol>
                <li>考生须携带本人学生证和身份证参加考试。</li>
                <li>考试开始15分钟后不得进入考场。</li>
                <li>严禁携带手机等通讯工具进入考场。</li>
                <li>遵守考场纪律，服从监考老师管理。</li>
            </ol>
            <p>请各学院及时通知学生，做好考试准备工作。</p>
            <p>教务处</p>
            <p>2025年6月1日</p>
        `,
        date: '2025-06-01',
        attachments: [
            { name: '考试安排表.xlsx', url: 'https://jwc.bwgl.cn/files/exam_schedule.xlsx' },
            { name: '考场规则.pdf', url: 'https://jwc.bwgl.cn/files/exam_rules.pdf' }
        ],
        url: 'https://jwc.bwgl.cn/announcement/20250601'
    },
    {
        title: '2025年暑期社会实践活动报名通知',
        content: `
            <p>各学院、全体同学：</p>
            <p>为深入贯彻落实党的二十大精神，引导广大青年学生在实践中受教育、长才干、作贡献，学校决定组织开展2025年暑期社会实践活动。现将有关事项通知如下：</p>
            <h3>一、活动主题</h3>
            <p>青春心向党·建功新时代</p>
            <h3>二、活动时间</h3>
            <p>2025年7月10日至8月20日</p>
            <h3>三、活动内容</h3>
            <ol>
                <li>乡村振兴实践</li>
                <li>科技支农服务</li>
                <li>教育关爱服务</li>
                <li>文化宣传服务</li>
                <li>疫情防控服务</li>
            </ol>
            <h3>四、报名方式</h3>
            <p>请各学院组织学生于6月15日前通过学校实践平台完成报名。</p>
            <p>联系人：张老师，联系电话：0771-12345678</p>
            <p>校团委</p>
            <p>2025年5月28日</p>
        `,
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