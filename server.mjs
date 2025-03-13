#!/usr/bin/env node

import express from "express";
import cors from "cors";
import { resolve } from "path";

const app = express();

// Get folder path from command line argument, or use current directory
const folderPath = resolve(process.argv[2] ?? ".");

const port = parseInt(process.argv[3] ?? "3000");

// Enable CORS for all routes
app.use(cors());

// Serve static files from the specified folder
app.use(express.static(folderPath));

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
  console.log(`Serving files from: ${folderPath}`);
});
