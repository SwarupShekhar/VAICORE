import subprocess

combined_setup_script = """# ─────────────────────────────────────────────────────────────
# Step 1: Wait for cloud-init (if still running) & update system
# ─────────────────────────────────────────────────────────────
cloud-init status --wait || true
apt-get update -qq && apt-get upgrade -y -qq

# ─────────────────────────────────────────────────────────────
# Step 2: Install Docker, docker-compose, git, nginx
# ─────────────────────────────────────────────────────────────
apt-get install -y -qq apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -qq
apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin git nginx certbot python3-certbot-nginx

# Enable & start Docker
systemctl enable docker
systemctl start docker

# ─────────────────────────────────────────────────────────────
# Step 3: Create 'deploy' user (non-root for Docker operations)
# ─────────────────────────────────────────────────────────────
useradd -m -s /bin/bash deploy
usermod -aG docker deploy

# ─────────────────────────────────────────────────────────────
# Step 3b: Configure SSH Key for 'deploy' user
# ─────────────────────────────────────────────────────────────
mkdir -p /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICl3iMBEsBuTQzbUzkonkyzEVOOim6IBfRbXzQwVq9uu swarupshekhar@Swarups-MacBook-Air.local" > /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh

# ─────────────────────────────────────────────────────────────
# Step 4: Create directory structure
# ─────────────────────────────────────────────────────────────
mkdir -p /opt/vaidikai-portal
chown deploy:deploy /opt/vaidikai-portal

# ─────────────────────────────────────────────────────────────
# Step 5: Configure firewall (UFW) — ONLY open required ports
#     IMPORTANT: Only ports 4005, 4006, 4007 + 22, 80, 443 are open
# ─────────────────────────────────────────────────────────────
ufw --force disable || true
ufw default deny incoming
ufw default allow outgoing

# Management & SSL
ufw allow 22/tcp     # SSH
ufw allow 80/tcp     # HTTP (for Let's Encrypt)
ufw allow 443/tcp    # HTTPS (for SSL)

# Application ports
ufw allow 4005/tcp   # Portal API
ufw allow 4006/tcp   # Label Studio UI
ufw allow 4007/tcp   # SAM ML Backend

ufw --force enable

echo ""
echo "✅ Firewall configured — only ports 22, 80, 443, 4005, 4006, 4007 are open"
ufw status verbose

# ─────────────────────────────────────────────────────────────
# Step 6: Create deployment helper script
# ─────────────────────────────────────────────────────────────
cat > /opt/vaidikai-portal/deploy.sh << 'EOF'
#!/bin/bash
cd /opt/vaidikai-portal/vaidikai-portal || exit 1

echo "=== VaidaikAI Deployment ==="

# Pull latest code
if [ -d .git ]; then
  echo "📦 Pulling latest code..."
  git pull origin main || echo "⚠️  Git pull failed — ensure repo is accessible"
fi

# Stop old containers
echo "🛑 Stopping old containers..."
docker-compose down || true

# Build new images
echo "🔨 Building new Docker images..."
docker-compose build --no-cache

# Start services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for startup
echo "⏳ Waiting 30 seconds for services..."
sleep 30

# Health checks
ERRORS=0
echo ""
echo "🔍 Health Checks:"

if curl -sf http://localhost:4005/health > /dev/null 2>&1; then
  echo "  ✅ Portal (port 4005): Healthy"
else
  echo "  ❌ Portal (port 4005): FAILED"
  ((ERRORS++))
fi

if curl -sf http://localhost:4006/health > /dev/null 2>&1; then
  echo "  ✅ Label Studio (port 4006): Healthy"
else
  echo "  ❌ Label Studio (port 4006): FAILED"
  ((ERRORS++))
fi

if curl -sf http://localhost:4007/health > /dev/null 2>&1; then
  echo "  ✅ SAM ML Backend (port 4007): Healthy"
else
  echo "  ❌ SAM ML Backend (port 4007): FAILED"
  ((ERRORS++))
fi

echo ""
if [ $ERRORS -eq 0 ]; then
  echo "🎉 All services deployed successfully!"
  echo ""
  IP=$(curl -s ifconfig.me)
  echo "Access URLs:"
  echo "  Portal:       http://$IP:4005"
  echo "  Label Studio: http://$IP:4006"
  echo "  SAM ML:       http://$IP:4007"
else
  echo "⚠️  Some services failed. Check logs:"
  echo "   cd /opt/vaidikai-portal/vaidikai-portal"
  echo "   docker-compose logs -f"
  exit 1
fi
EOF

chmod +x /opt/vaidikai-portal/deploy.sh
chown deploy:deploy /opt/vaidikai-portal/deploy.sh

echo ""
echo "=== Vultr Server Ready ==="
echo ""
echo "Next steps:"
echo ""
echo "1️⃣  Switch to deploy user on server:"
echo "   su - deploy"
echo ""
echo "2️⃣  Clone your repository (replace with your actual GitHub URL):"
echo "   git clone https://github.com/YOUR_USERNAME/vaidikai-portal.git /opt/vaidikai-portal/vaidikai-portal"
echo ""
echo "3️⃣  Navigate to project:"
echo "   cd /opt/vaidikai-portal/vaidikai-portal"
echo ""
echo "4️⃣  Create .env file (copy from your local machine):"
echo "   nano .env"
echo "   (paste your production environment variables)"
echo ""
echo "5️⃣  Deploy:"
echo "   /opt/vaidikai-portal/deploy.sh"
echo ""
echo "Or run directly:"
echo "   docker-compose up -d"
echo ""
echo "────────────────────────────────────────────────"
echo "Server IP: $(curl -s ifconfig.me)"
echo "Firewall: ONLY ports 22, 80, 443, 4005, 4006, 4007 open"
echo "SSH Authorized for 'deploy' user!"
echo "────────────────────────────────────────────────"
"""

# Copy to macOS clipboard
p = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
p.communicate(combined_setup_script.encode('utf-8'))
print("✅ Copied combined setup script (including SSH Key config) to clipboard!")
