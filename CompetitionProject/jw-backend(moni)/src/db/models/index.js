const { DataTypes } = require('sequelize');
const { sequelize } = require('../index');

const Student = sequelize.define('Student', {
    studentId: { type: DataTypes.STRING(100), primaryKey: true },
    name: DataTypes.STRING(500),
    gender: DataTypes.STRING(200),
    enrollmentYear: DataTypes.STRING(500),
    className: DataTypes.STRING(500),
    major: DataTypes.STRING(500),
    college: DataTypes.STRING(500),
    lastSync: DataTypes.DATE
}, {
    tableName: 'Student',
    timestamps: false
});

const Course = sequelize.define('Course', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100) },
    semester: DataTypes.STRING(100),
    name: DataTypes.STRING(1000),
    dayOfWeek: DataTypes.STRING(50),
    week: DataTypes.INTEGER,
    period: DataTypes.STRING(200),
    teacher: DataTypes.STRING(500),
    weeks: DataTypes.STRING(500),
    location: DataTypes.STRING(1000),
    courseType: DataTypes.STRING(500),
    raw: DataTypes.TEXT('long')
}, {
    tableName: 'Course',
    timestamps: false
});

const Grade = sequelize.define('Grade', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100) },
    semester: DataTypes.STRING(100),
    courseCode: DataTypes.STRING(200),
    courseName: DataTypes.STRING(1000),
    score: DataTypes.STRING(500),
    credit: DataTypes.STRING(200),
    gradePoint: DataTypes.STRING(200),
    courseType: DataTypes.STRING(500),
    examType: DataTypes.STRING(500)
}, {
    tableName: 'Grade',
    timestamps: false
});

const Exam = sequelize.define('Exam', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100) },
    courseName: DataTypes.STRING(1000),
    examTime: DataTypes.STRING(500),
    location: DataTypes.STRING(1000),
    seatNumber: DataTypes.STRING(500),
    examType: DataTypes.STRING(500),
    status: DataTypes.STRING(500)
}, {
    tableName: 'Exam',
    timestamps: false
});

const Plan = sequelize.define('Plan', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100) },
    semester: DataTypes.STRING(100),
    courseCode: DataTypes.STRING(200),
    courseName: DataTypes.STRING(1000),
    teachingUnit: DataTypes.STRING(500),
    credit: DataTypes.STRING(200),
    totalHours: DataTypes.STRING(200),
    courseType: DataTypes.STRING(500),
    courseAttribute: DataTypes.STRING(500),
    examType: DataTypes.STRING(500),
    isExam: DataTypes.STRING(200)
}, {
    tableName: 'Plan',
    timestamps: false
});

const Progress = sequelize.define('Progress', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100) },
    category: DataTypes.STRING(2000),
    requiredCredits: DataTypes.STRING(500),
    completedCredits: DataTypes.STRING(500),
    currentCredits: DataTypes.STRING(500),
    remainingCredits: DataTypes.STRING(500)
}, {
    tableName: 'Progress',
    timestamps: false
});

const ElectricityReminder = sequelize.define('ElectricityReminder', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100), unique: true },
    enabled: { type: DataTypes.BOOLEAN, defaultValue: false },
    threshold: { type: DataTypes.FLOAT, defaultValue: 10 },
    electricityAccount: DataTypes.STRING(500),
    electricityPassword: DataTypes.STRING(500),
    roomId: DataTypes.STRING(200),
    campusId: DataTypes.STRING(200),
    buildingId: DataTypes.STRING(200),
    createdAt: DataTypes.DATE,
    updatedAt: DataTypes.DATE
}, {
    tableName: 'ElectricityReminder',
    timestamps: true
});

const UserPushToken = sequelize.define('UserPushToken', {
    id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
    studentId: { type: DataTypes.STRING(100), unique: true },
    pushToken: { type: DataTypes.STRING(500), allowNull: false },
    deviceInfo: DataTypes.STRING(500),
    isActive: { type: DataTypes.BOOLEAN, defaultValue: true },
    createdAt: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
    lastActiveAt: { type: DataTypes.DATE, defaultValue: DataTypes.NOW }
}, {
    tableName: 'UserPushToken',
    timestamps: false
});

module.exports = {
    Student,
    Course,
    Grade,
    Exam,
    Plan,
    Progress,
    ElectricityReminder,
    UserPushToken
};
