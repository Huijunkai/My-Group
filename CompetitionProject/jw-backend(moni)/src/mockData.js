const fs = require('fs');
const path = require('path');
const DATA_DIR = path.join(__dirname, '../data');

function readCsvAsJson(filename) {
    const filePath = path.join(DATA_DIR, filename);
    try {
        if (!fs.existsSync(filePath)) {
            console.warn(`[Mock Data] 警告: 找不到文件 ${filePath}，返回空数组。`);
            return [];
        }
        
        const fileContent = fs.readFileSync(filePath, 'utf8');
        const lines = fileContent.trim().split(/\r?\n/);
        if (lines.length < 2) return [];
        
        const headers = lines[0].replace(/^\uFEFF/, '').split(',');
        const dataList = [];
        
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',');
            const record = {};
            headers.forEach((header, index) => {
                record[header] = values[index];
            });
            dataList.push(record);
        }
        return dataList;
    } catch (error) {
        console.error(`[Mock Data] 加载 ${filename} 失败:`, error.message);
        return [];
    }
}

console.log('[Mock Data] 正在从 CSV 文件加载初始化数据...');


const mockStudents = readCsvAsJson('students.csv');
const mockTimetable = readCsvAsJson('courses.csv');
const mockExams = readCsvAsJson('exam.csv');
const mockProgress = readCsvAsJson('progress.csv');
const rawGrades = readCsvAsJson('grades.csv');
const mockGrades = {};
rawGrades.forEach(g => {
    if (!mockGrades[g.semester]) mockGrades[g.semester] = [];
    mockGrades[g.semester].push(g);
});


const rawPlans = readCsvAsJson('plans.csv');
const mockPlans = {};
rawPlans.forEach(p => {
    if (!mockPlans[p.semester]) mockPlans[p.semester] = [];
    mockPlans[p.semester].push(p);
});

console.log('[Mock Data] 成功从 CSV 文件加载所有模拟数据！');

//1111
module.exports = {
    mockStudents,
    mockTimetable,
    mockGrades,
    mockExams,
    mockPlans,
    mockProgress
};