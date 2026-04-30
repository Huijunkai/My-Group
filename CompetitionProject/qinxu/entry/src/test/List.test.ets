import localUnitTest from './LocalUnit.test';
import dataStoreTest from './DataStore.test';

export default function testsuite() {
  localUnitTest();
  dataStoreTest();
}