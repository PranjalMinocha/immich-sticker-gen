System-Level Object Store Mount (Docker-Compatible)

Use this when Docker cannot bind a user-session FUSE mount (for example `/home/cc/my_object_store`).

1. Copy `rclone-objstore.service.example` to `/etc/systemd/system/rclone-objstore.service`.
2. Update the rclone remote name (`objstore-proj28`) if needed.
3. Ensure `/etc/fuse.conf` contains `user_allow_other`.
4. Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rclone-objstore.service
```

5. Verify:

```bash
mount | grep /mnt/objstore
docker run --rm -v /mnt/objstore:/data:ro alpine ls /data
```

Use `/mnt/objstore` as `OBJSTORE_DATA_ROOT` for retraining jobs.
