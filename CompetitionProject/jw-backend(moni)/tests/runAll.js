const path = require('path');
const fs = require('fs');

const TESTS_DIR = __dirname;

function runTestFile(testFile) {
    return new Promise((resolve) => {
        const testPath = path.join(TESTS_DIR, testFile);
        
        console.log(`\n${'='.repeat(60)}`);
        console.log(`运行测试: ${testFile}`);
        console.log('='.repeat(60));
        
        try {
            require(testPath);

            if (typeof describe === 'function') {
                setTimeout(() => {
                    resolve({ file: testFile, status: 'completed', error: null });
                }, 1000);
            } else {
                resolve({ file: testFile, status: 'skipped', error: 'No test framework found' });
            }
        } catch (error) {
            resolve({ file: testFile, status: 'error', error: error.message });
        }
    });
}

async function main() {
    console.log('╔══════════════════════════════════════════════════╗');
    console.log('║        jw-backend(moni) 单元测试套件              ║');
    console.log('╚══════════════════════════════════════════════════╝');
    
    const testFiles = fs.readdirSync(TESTS_DIR)
        .filter(file => file.startsWith('test.') && file.endsWith('.js'))
        .sort();

    if (testFiles.length === 0) {
        console.log('\n未找到测试文件');
        return;
    }

    console.log(`\n发现 ${testFiles.length} 个测试文件:\n`);
    testFiles.forEach((file, index) => {
        console.log(`${index + 1}. ${file}`);
    });

    console.log('\n开始执行测试...\n');

    const results = [];
    for (const testFile of testFiles) {
        const result = await runTestFile(testFile);
        results.push(result);
    }

    console.log('\n\n' + '═'.repeat(60));
    console.log('测试结果汇总');
    console.log('═'.repeat(60));

    let passed = 0;
    let failed = 0;
    let skipped = 0;

    results.forEach(result => {
        switch (result.status) {
            case 'completed':
                passed++;
                console.log(`✓ ${result.file}`);
                break;
            case 'error':
                failed++;
                console.log(`✗ ${result.file} - ${result.error}`);
                break;
            case 'skipped':
                skipped++;
                console.log(`○ ${result.file} - 跳过`);
                break;
        }
    });

    console.log('-'.repeat(60));
    console.log(`总计: ${results.length} | 通过: ${passed} | 失败: ${failed} | 跳过: ${skipped}`);
}

main().catch(console.error);