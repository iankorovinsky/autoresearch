import assert from 'node:assert/strict';
import { test } from 'node:test';
import { authDestination } from '../src/lib/auth/navigation.ts';
import { credentials, validEmail, validPassword } from '../src/lib/auth/validation.ts';

test('OAuth return paths cannot escape the application or select arbitrary routes', () => {
  for (const value of [null, '', '/', 'https://example.com', '//example.com', '/\\example.com', 'javascript:alert(1)', '/admin', '/auth/reset-password?next=https://example.com', '%2f%2fexample.com']) {
    assert.equal(authDestination(value), '/');
  }
  assert.equal(authDestination('/auth/reset-password'), '/auth/reset-password');
});
test('credential parsing preserves passwords and trims email only', () => {
  const form = new FormData();
  form.set('email', ' researcher@example.com ');
  form.set('password', ' leading and trailing ');
  assert.deepEqual(credentials(form), {email:'researcher@example.com', password:' leading and trailing '});
});
test('signup and reset validate the password policy on the server', () => {
  assert.equal(validPassword('x'.repeat(11)), false);
  assert.equal(validPassword('x'.repeat(12)), true);
  assert.equal(validPassword('x'.repeat(128)), true);
  assert.equal(validPassword('x'.repeat(129)), false);
});
test('email validation rejects malformed and oversized input', () => {
  assert.equal(validEmail('researcher@example.com'), true);
  for (const email of ['', 'no-at-sign', 'a@', 'a@b', 'a b@example.com', `${'x'.repeat(250)}@example.com`]) {
    assert.equal(validEmail(email), false);
  }
});
