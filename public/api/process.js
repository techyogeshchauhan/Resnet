const { put } = require('@vercel/blob');
const ffmpeg = require('ffmpeg-static');
const { exec } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const os = require('os');

module.exports = async (req, res) => {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    try {
        const { name, dob } = req.body;
        const videoFile = req.files.video;
        const tempDir = os.tmpdir();
        const videoPath = path.join(tempDir, `${name}_${dob}.webm`);
        const outputDir = path.join(tempDir, `${name}_${dob}_frames`);
        await fs.mkdir(outputDir, { recursive: true });

        // Save uploaded video temporarily
        await fs.writeFile(videoPath, videoFile.data);

        // Extract frames using FFmpeg
        const framePattern = path.join(outputDir, 'frame_%04d.jpg');
        await new Promise((resolve, reject) => {
            exec(
                `${ffmpeg} -i ${videoPath} -vf fps=1 ${framePattern} -y`,
                (err) => err ? reject(err) : resolve()
            );
        });

        // Upload frames to Vercel Blob
        const frameFiles = (await fs.readdir(outputDir)).filter(f => f.endsWith('.jpg'));
        for (const frame of frameFiles) {
            const framePath = path.join(outputDir, frame);
            const blob = await put(`dataset/${name}_${dob}/${frame}`, await fs.readFile(framePath), {
                access: 'public'
            });
        }

        // Clean up
        await fs.rm(outputDir, { recursive: true });
        await fs.unlink(videoPath);

        res.status(200).json({ message: `Data for ${name} saved successfully.` });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: `Processing failed: ${error.message}` });
    }
};