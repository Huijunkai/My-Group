const cheerio = require('cheerio');

/**
 * 解析学生个人信息
 */
function parseStudentInfo(html) {
    const $ = cheerio.load(html);
    const studentInfo = {
        name: '',
        gender: '',
        enrollmentYear: '',
        className: '',
        major: '',
        college: ''
    };

    const $table = $('#xjkpTable');
    if ($table.length === 0) return null;

    studentInfo.name = $table.find('td:contains("姓名")').first().next('td').text().replace(/\s+/g, '');
    studentInfo.gender = $table.find('td:contains("性别")').first().next('td').text().replace(/\s+/g, '');

    $table.find('td').each((i, el) => {
        const text = $(el).text().trim();
        if (text.includes('院系：')) {
            studentInfo.college = text.split('：')[1].trim();
        } else if (text.includes('专业：')) {
            studentInfo.major = text.split('：')[1].trim();
        } else if (text.includes('班级：')) {
            studentInfo.className = text.split('：')[1].trim();
        }
    });

    studentInfo.enrollmentYear = $table.find('td:contains("入学日期")').next('td').text().replace(/\s+/g, '');

    return studentInfo;
}

/**
 * 解析课表信息
 */
function parseTimetable(html) {
    const $ = cheerio.load(html);
    const courses = [];

    // 学期字段在不同学校/版本里 id 不一致：常见是 xnxq01id / xnxqh
    // 兜底：取页面第一个 selected 的 option（一般就是学年学期下拉的选中项）
    let semester =
        $('#xnxq01id option:selected').first().text().trim() ||
        $('#xnxqh option:selected').first().text().trim() ||
        $('.Nsb_right_title_sj').text().trim() ||
        $('option[selected]').first().text().trim() ||
        '未知学期';
    
    const $table = $('#kbtable');

    // 解析工具：把 div 内 HTML 变成“带换行”的纯文本行
    const htmlToLines = (innerHtml) => {
        if (!innerHtml) return [];
        const withNewlines = String(innerHtml)
            // 关键：强智系统大量使用 <br/>，不能只 split("<br>")
            .replace(/<br\s*\/?>/gi, '\n')
            .replace(/&nbsp;/gi, ' ');
        const text = cheerio.load(`<div>${withNewlines}</div>`).text();
        return text
            .split('\n')
            .map(s => String(s).trim())
            .filter(Boolean);
    };

    // 周次规范化（只保留数字/逗号/短横线；保留“全部”但去掉括号）
    const normalizeWeeks = (input) => {
        if (!input) return '';
        let s = String(input).replace(/\s+/g, '');
        s = s.replace(/（/g, '(').replace(/）/g, ')');
        s = s.replace(/[，、]/g, ',');
        s = s.replace(/[～—–－]/g, '-');
        s = s.replace(/至/g, '-');
        s = s.replace(/\[?\d{1,2}-\d{1,2}节\]?/g, '');
        s = s.replace(/周/g, '');
        let hasAll = false;
        s = s.replace(/\((.*?)\)/g, (_m, inner) => {
            if (String(inner).includes('全部')) {
                hasAll = true;
                return '全部';
            }
            return '';
        });
        s = s.replace(/第/g, '').replace(/单/g, '').replace(/双/g, '');
        const m = s.match(/[0-9]{1,2}(?:-[0-9]{1,2})?(?:,[0-9]{1,2}(?:-[0-9]{1,2})?)*/);
        if (m && m[0]) return m[0] + (hasAll ? '全部' : '');
        if (hasAll) return '全部';
        return s.replace(/[^0-9,\-全都部,]/g, '');
    };

    // 将周次表达式拆分为单周数组（不做单双周推断，仅按范围/逗号展开）
    const parseWeekList = (expr) => {
        if (!expr) return [];
        const s = String(expr).replace(/全部/g, '').replace(/[^0-9,\-]/g, '');
        if (!s) return [];
        const out = [];
        for (const part of s.split(',').filter(Boolean)) {
            if (part.includes('-')) {
                const [a, b] = part.split('-');
                const start = parseInt(a, 10);
                const end = parseInt(b, 10);
                if (!Number.isNaN(start) && !Number.isNaN(end)) {
                    const lo = Math.min(start, end);
                    const hi = Math.max(start, end);
                    for (let w = lo; w <= hi; w++) out.push(w);
                }
            } else {
                const w = parseInt(part, 10);
                if (!Number.isNaN(w)) out.push(w);
            }
        }
        return Array.from(new Set(out)).sort((x, y) => x - y);
    };

    // 遍历每个格子：优先取详细版 kbcontent（包含老师/节次），没有再取 kbcontent1
    $table.find('td').each((_, td) => {
        const $td = $(td);
        const $detail = $td.find('div.kbcontent').first();
        const $simple = $td.find('div.kbcontent1').first();
        const $contentEl = $detail.length ? $detail : $simple;
        if (!$contentEl.length) return;

        const content = $contentEl.html() || '';
        if (!content.trim() || content.trim() === '&nbsp;') return;

        // 通过单元格在行中的索引来确定星期几（index 1 是周一）
        const columnIndex = $td.index();
        const weekDays = ['', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];
        const dayOfWeek = weekDays[columnIndex] || '未知';

        const lines = htmlToLines(content);
        if (lines.length === 0) return;

        // 按分隔线拆分（同一格可能有多门课）
        const chunks = [];
        let current = [];
        for (const line of lines) {
            if (/^-{5,}$/.test(line)) {
                if (current.length > 0) chunks.push(current);
                current = [];
                continue;
            }
            current.push(line);
        }
        if (current.length > 0) chunks.push(current);

        for (const chunkLines of chunks) {
            if (!chunkLines || chunkLines.length < 2) continue;
            if (chunkLines.length === 1 && chunkLines[0] === '&nbsp;') continue;

            // 强智详细版常见结构：
            // 0: 课程名[学时][性质]
            // 1: 老师
            // 2: 班级/周次/节次（包含 [01-02节]）
            // 3: 上课地点
            const rawText = chunkLines.join(' | ');
            const titleLine = chunkLines[0] || '';
            const name = String(titleLine).replace(/\[.*?\]/g, '').trim() || titleLine.trim();

            // 找到“包含节次”的那一行作为时间行
            let timeLineIndex = chunkLines.findIndex(l => /\[\d{1,2}-\d{1,2}节\]/.test(l) || /(\d{1,2})-(\d{1,2})节/.test(l));
            if (timeLineIndex === -1) {
                // 兜底：存在但格式不含“节]”
                timeLineIndex = chunkLines.findIndex(l => (l.includes('周') || l.includes('节') || (l.includes('[') && l.includes(']'))));
            }
            const weekStr = timeLineIndex !== -1 ? (chunkLines[timeLineIndex] || '') : '';
            const timeLine = String(weekStr).replace(/\s+/g, '');

            // 节次
            let periodStr = '';
            const periodBracketMatch = timeLine.match(/\[(\d{1,2})-(\d{1,2})节\]/);
            if (periodBracketMatch) {
                periodStr = `${periodBracketMatch[1]}-${periodBracketMatch[2]}节`;
            }
            const periodMatch = weekStr.match(/(\d{1,2})-(\d{1,2})节/);
            if (periodMatch) {
                periodStr = periodMatch[0];
            }

            // 周次（剔除节次残留）
            let weeksOnlySource = timeLine;
            if (periodBracketMatch) weeksOnlySource = weeksOnlySource.replace(periodBracketMatch[0], '');
            if (periodStr) weeksOnlySource = weeksOnlySource.replace(periodStr.replace('节', ''), '');
            const weekExpr = normalizeWeeks(weeksOnlySource);
            const weekList = parseWeekList(weekExpr);

            // 老师：详细版通常第 2 行；简单版可能没有老师，尽量取“非时间行/非地点行”的一行
            let teacher = '';
            if (chunkLines.length >= 2) {
                teacher = String(chunkLines[1]).replace(/^老师[:：]?\s*/g, '').trim();
            }

            // 上课地点：优先取最后一行；如果最后一行看起来像“班级/周次/节次”，再往后找
            let location = chunkLines[chunkLines.length - 1] || '未知';
            if (location === weekStr && chunkLines.length >= 3) {
                location = chunkLines[chunkLines.length - 2] || '未知';
            }
            location = String(location).replace(/^上课地点[:：]?\s*/g, '').trim() || '未知';

            const base = {
                semester,
                dayOfWeek,
                name,
                teacher,
                period: periodStr,
                location,
                raw: rawText
            };

            // 按周拆分存储：每条记录对应一个 week
            if (weekList.length > 0) {
                for (const weekNum of weekList) {
                    courses.push({
                        ...base,
                        week: weekNum,
                        weeks: String(weekNum)
                    });
                }
            } else {
                courses.push({
                    ...base,
                    week: 0,
                    weeks: '0'
                });
            }
        }
    });

    return courses;
}

/**
 * 解析成绩信息
 */
function parseGrades(html) {
    const $ = cheerio.load(html);
    const gradesGrouped = {};

    let $table = $('#dataList');
    if ($table.length === 0) {
        $table = $('table').filter((i, el) => $(el).text().includes('成绩') && $(el).find('tr').length > 1);
    }
    
    $table.find('tr').each((i, el) => {
        const tds = $(el).find('td');
        const firstTdText = $(tds[0]).text().trim();
        if (!firstTdText || firstTdText === '序号' || firstTdText.includes('课程')) return;

        if (tds.length >= 6) {
            const semester = $(tds[1]).text().trim();
            const gradeItem = {
                courseCode: $(tds[2]).text().trim(),
                courseName: $(tds[3]).text().trim(),
                score: $(tds[4]).text().trim(),
                credit: $(tds[5]).text().trim(),
                gradePoint: $(tds[6]).text().trim(),
                courseType: $(tds[7]).text().trim(),
                examType: $(tds[8]).text().trim()
            };

            if (!gradesGrouped[semester]) {
                gradesGrouped[semester] = [];
            }
            gradesGrouped[semester].push(gradeItem);
        }
    });

    return gradesGrouped;
}

/**
 * 解析考试安排
 */
function parseExams(html) {
    const $ = cheerio.load(html);
    const exams = [];
    const $table = $('#dataList');
    
    $table.find('tr').each((i, el) => {
        if (i === 0) return;
        const tds = $(el).find('td');
        if (tds.length >= 6) {
            exams.push({
                courseName: $(tds[1]).text().trim(),
                examTime: $(tds[3]).text().trim(),
                location: $(tds[4]).text().trim(),
                seatNumber: $(tds[5]).text().trim(),
                examType: $(tds[2]).text().trim(),
                status: $(tds[6]).text().trim()
            });
        }
    });

    return exams;
}

/**
 * 解析学期计划
 */
function parseSemesterPlan(html) {
    const $ = cheerio.load(html);
    const plansGrouped = {};
    const $table = $('#dataList');
    
    $table.find('tr').each((i, el) => {
        if (i === 0) return;
        const tds = $(el).find('td');
        if (tds.length >= 6) {
            const semester = $(tds[1]).text().trim();
            const planItem = {
                courseCode: $(tds[2]).text().trim(),
                courseName: $(tds[3]).text().trim(),
                credit: $(tds[4]).text().trim(),
                totalHours: $(tds[5]).text().trim(),
                courseType: $(tds[6]).text().trim(),
                examType: $(tds[7]).text().trim()
            };

            if (!plansGrouped[semester]) {
                plansGrouped[semester] = [];
            }
            plansGrouped[semester].push(planItem);
        }
    });

    return plansGrouped;
}

/**
 * 解析学习进度
 */
function parseStudyProgress(html) {
    const $ = cheerio.load(html);
    const progressData = [];

    const $table = $('table').filter((i, el) => {
        const headText = $(el).find('tr').first().text();
        return headText.includes('课程体系') && headText.includes('要求学分');
    }).first();
    
    $table.find('tr').each((i, el) => {
        const tds = $(el).find('td');
        if (tds.length < 5) return;
        const category = $(tds[0]).text().trim();
        if (!category || category === '课程体系(属性)') return;

        progressData.push({
            category: category,
            requiredCredits: $(tds[1]).text().trim(),
            completedCredits: $(tds[2]).text().trim(),
            currentCredits: $(tds[3]).text().trim(),
            remainingCredits: $(tds[4]).text().trim()
        });
    });

    return progressData;
}

module.exports = {
    parseStudentInfo,
    parseTimetable,
    parseGrades,
    parseExams,
    parseSemesterPlan,
    parseStudyProgress
};
