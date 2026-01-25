const { DataTypes } = require('sequelize');
const { sequelize } = require('../index');

// 学生基本信息表
const Student = sequelize.define('Student', {
    studentId: { type: DataTypes.STRING, primaryKey: true }, // 学号作为主键
    name: DataTypes.STRING,
    gender: DataTypes.STRING,
    enrollmentYear: DataTypes.STRING,
    className: DataTypes.STRING,
    major: DataTypes.STRING,
    college: DataTypes.STRING,
    lastSync: { type: DataTypes.DATE, defaultValue: DataTypes.NOW }
});

// 课程表
const Course = sequelize.define('Course', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    semester: { type: DataTypes.STRING, primaryKey: true },
    name: { type: DataTypes.STRING, primaryKey: true },
    dayOfWeek: { type: DataTypes.STRING, primaryKey: true },
    teacher: DataTypes.STRING,
    weeks: DataTypes.STRING,
    location: DataTypes.STRING,
    raw: DataTypes.TEXT
});

// 成绩表
const Grade = sequelize.define('Grade', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    semester: { type: DataTypes.STRING, primaryKey: true },
    courseCode: { type: DataTypes.STRING, primaryKey: true },
    courseName: DataTypes.STRING,
    score: DataTypes.STRING,
    credit: DataTypes.STRING,
    gradePoint: DataTypes.STRING,
    courseType: DataTypes.STRING,
    examType: DataTypes.STRING
});

// 考试安排表
const Exam = sequelize.define('Exam', {
    studentId: { type: DataTypes.STRING, primaryKey: true },
    courseName: { type: DataTypes.STRING, primaryKey: true },
    examTime: { type: DataTypes.STRING, primaryKey: true },
    location: DataTypes.STRING,
    seatNumber: DataTypes.STRING,
    examType: DataTypes.STRING,
    status: DataTypes.STRING
});

module.exports = {
    Student,
    Course,
    Grade,
    Exam
};
