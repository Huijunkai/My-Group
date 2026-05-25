const { encrypt, decrypt, encryptStudentInfo, decryptStudentInfo, getEncryptionKeyBase64, getIvBase64 } = require('../src/utils/encryption');

console.log('=== 加密解密测试 ===\n');

console.log('1. 测试基本加密解密:');
const originalText = '张三';
const encrypted = encrypt(originalText);
const decrypted = decrypt(encrypted);
console.log(`原文: ${originalText}`);
console.log(`加密后: ${encrypted}`);
console.log(`解密后: ${decrypted}`);
console.log(`验证: ${originalText === decrypted ? '✓ 成功' : '✗ 失败'}\n`);

console.log('2. 测试学生信息加密:');
const studentInfo = {
    name: '李四',
    gender: '男',
    enrollmentYear: '2021',
    className: '计算机科学2101班',
    major: '计算机科学与技术',
    college: '信息工程学院'
};
const encryptedInfo = encryptStudentInfo(studentInfo);
console.log('原始信息:', JSON.stringify(studentInfo, null, 2));
console.log('加密后:', JSON.stringify(encryptedInfo, null, 2));
const decryptedInfo = decryptStudentInfo(encryptedInfo);
console.log('解密后:', JSON.stringify(decryptedInfo, null, 2));
console.log(`验证: ${JSON.stringify(studentInfo) === JSON.stringify(decryptedInfo) ? '✓ 成功' : '✗ 失败'}\n`);

console.log('3. 测试空值和特殊值:');
console.log(`空字符串: "${encrypt('')}" -> "${decrypt(encrypt(''))}"`);
console.log(`null: ${encrypt(null)} -> ${decrypt(encrypt(null))}`);
console.log(`undefined: ${encrypt(undefined)} -> ${decrypt(encrypt(undefined))}`);
console.log(`数字: ${encrypt('123.5')} -> ${decrypt(encrypt('123.5'))}\n`);

console.log('4. 加密密钥信息:');
console.log(`Key (Base64): ${getEncryptionKeyBase64()}`);
console.log(`IV (Base64): ${getIvBase64()}\n`);

console.log('=== 测试完成 ===');
