const { DataTypes } = require('sequelize');
const { sequelize } = require('../index');

// 学生基本信息表
const Student = sequelize.define('Student', {
    studentId: { type: DataTypes.STRING(50), primaryKey: true }, // 学号作为主键
    name: DataTypes.STRING,
    gender: DataTypes.STRING,
    enrollmentYear: DataTypes.STRING,
    className: DataTypes.STRING,
    major: DataTypes.STRING,
    college: DataTypes.STRING,
    lastSync: { type: DataTypes.DATE, defaultValue: DataTypes.NOW }
}, {
    tableName: 'Student',  // 明确指定表名为单数形式
    timestamps: false   // 不自动添加 createdAt 和 updatedAt 字段
});

// 课程表
const Course = sequelize.define('Course', {
    // 说明：
    // - 必须把 week + period 纳入主键，否则同名课程在不同周/不同节会被覆盖
    studentId: { type: DataTypes.STRING(50), primaryKey: true },
    semester: { type: DataTypes.STRING(50), primaryKey: true },
    name: { type: DataTypes.STRING(100), primaryKey: true },
    dayOfWeek: { type: DataTypes.STRING(20), primaryKey: true },
    week: { type: DataTypes.INTEGER, primaryKey: true }, // 单周：1,2,3...
    period: { type: DataTypes.STRING(50), primaryKey: true }, // 节次字符串，如 "01-02节"
    teacher: DataTypes.STRING,
    weeks: DataTypes.STRING, // 存储单周（与 week 对齐），例如 "6"
    location: DataTypes.STRING,
    courseType: DataTypes.STRING, // 课程类型：必修/选修
    // 兼容字段：前端仍可能用 raw 做解析兜底
    raw: DataTypes.TEXT
}, {
    tableName: 'Course',  // 明确指定表名为单数形式
    timestamps: false   // 不自动添加 createdAt 和 updatedAt 字段
});

// 成绩表
const Grade = sequelize.define('Grade', {
    studentId: { type: DataTypes.STRING(50), primaryKey: true },
    semester: { type: DataTypes.STRING(50), primaryKey: true },
    courseCode: { type: DataTypes.STRING(50), primaryKey: true },
    courseName: DataTypes.STRING,
    score: DataTypes.STRING,
    credit: DataTypes.STRING,
    gradePoint: DataTypes.STRING,
    courseType: DataTypes.STRING,
    examType: DataTypes.STRING
}, {
    tableName: 'Grade',  // 明确指定表名为单数形式
    timestamps: false   // 不自动添加 createdAt 和 updatedAt 字段
});

// 考试安排表
const Exam = sequelize.define('Exam', {
    studentId: { type: DataTypes.STRING(50), primaryKey: true },
    courseName: { type: DataTypes.STRING(100), primaryKey: true },
    examTime: { type: DataTypes.STRING(50), primaryKey: true },
    location: DataTypes.STRING,
    seatNumber: DataTypes.STRING,
    examType: DataTypes.STRING,
    status: DataTypes.STRING
}, {
    tableName: 'Exam',  // 明确指定表名为单数形式
    timestamps: false   // 不自动添加 createdAt 和 updatedAt 字段
});

// 学期计划表
const Plan = sequelize.define('Plan', {
    studentId: { type: DataTypes.STRING(50), primaryKey: true },
    semester: { type: DataTypes.STRING(50), primaryKey: true },
    courseCode: { type: DataTypes.STRING(50), primaryKey: true },
    courseName: DataTypes.STRING,
    teachingUnit: DataTypes.STRING,
    credit: DataTypes.STRING,
    totalHours: DataTypes.STRING,
    examType: DataTypes.STRING,
    courseAttribute: DataTypes.STRING,
    isExam: DataTypes.STRING
}, {
    tableName: 'Plan',
    timestamps: false
});

// 学习进度表
const Progress = sequelize.define('Progress', {
    studentId: { type: DataTypes.STRING(50), primaryKey: true },
    category: { type: DataTypes.STRING(50), primaryKey: true }, // 课程体系
    requiredCredits: DataTypes.STRING,
    completedCredits: DataTypes.STRING,
    currentCredits: DataTypes.STRING,
    remainingCredits: DataTypes.STRING
}, {
    tableName: 'Progress',  // 明确指定表名为单数形式
    timestamps: false   // 不自动添加 createdAt 和 updatedAt 字段
});

// 电费提醒设置表
const ElectricityReminder = sequelize.define('ElectricityReminder', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(50), unique: true },
    enabled: { type: DataTypes.BOOLEAN, defaultValue: false },
    threshold: { type: DataTypes.FLOAT, defaultValue: 10 },
    electricityAccount: DataTypes.STRING(50),
    roomId: DataTypes.STRING(50),
    campusId: DataTypes.STRING(50),
    buildingId: DataTypes.STRING(50),
    createdAt: DataTypes.DATE,
    updatedAt: DataTypes.DATE
}, {
    tableName: 'ElectricityReminder',
    timestamps: true
});

module.exports = {
    Student,
    Course,
    Grade,
    Exam,
    Plan,
    Progress,
    ElectricityReminder
};