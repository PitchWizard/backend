// frontend/src/api/authApi.js
import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";

export async function login(username, password) {
  const form = new FormData();
  form.append("username", username);
  form.append("password", password);
  const res = await axios.post(`${BASE_URL}/login`, form);
  return res.data; // UserOut
}

export async function signup({ username, email, password }) {
  const res = await axios.post(`${BASE_URL}/users`, {
    username,
    email,
    password,
  });
  return res.data; // UserOut
}
