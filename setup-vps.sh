#!/bin/bash

# DistilBERT Service Setup Script for VPS
# This script sets up the DistilBERT FastAPI service on your VPS

set -e  # Exit on any error

echo "🚀 Setting up DistilBERT Service on VPS..."

# Configuration
SERVICE_NAME="distilbert-service"
SERVICE_DIR="/opt/chatbot/services/${SERVICE_NAME}"
SERVICE_USER="distilbert"
PYTHON_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "Please run this script as root or with sudo"
        exit 1
    fi
}

# Function to update system
update_system() {
    print_status "Updating system packages..."
    apt update && apt upgrade -y
    print_success "System packages updated"
}

# Function to install Python 3.9
install_python() {
    print_status "Installing Python ${PYTHON_VERSION}..."
    
    # Install Python and pip
    apt install -y python3-venv python3-pip python3-dev

    
    # Install build tools
    apt install -y build-essential gcc g++ curl wget git
    
    # Update alternatives to use Python 3.9 as default
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
    update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip${PYTHON_VERSION} 1
    
    print_success "Python ${PYTHON_VERSION} installed successfully"
}

# Function to create service user
create_service_user() {
    print_status "Creating service user: ${SERVICE_USER}..."
    
    if id "$SERVICE_USER" &>/dev/null; then
        print_warning "User ${SERVICE_USER} already exists"
    else
        useradd -r -s /bin/bash -m -d /home/${SERVICE_USER} ${SERVICE_USER}
        print_success "User ${SERVICE_USER} created"
    fi
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Create main directories
    mkdir -p ${SERVICE_DIR}
    mkdir -p ${SERVICE_DIR}/models/cache
    mkdir -p ${SERVICE_DIR}/logs
    mkdir -p /var/log/${SERVICE_NAME}
    
    # Set permissions
    chown -R ${SERVICE_USER}:${SERVICE_USER} ${SERVICE_DIR}
    chown -R ${SERVICE_USER}:${SERVICE_USER} /var/log/${SERVICE_NAME}
    
    print_success "Directory structure created"
}

# Function to install service files
install_service_files() {
    print_status "Installing service files..."
    
    # Copy all service files to the target directory
    if [ -d "/tmp/distilbert-service" ]; then
        cp -r /tmp/distilbert-service/* ${SERVICE_DIR}/
    else
        print_error "Service files not found in /tmp/distilbert-service"
        print_status "Please upload your service files to /tmp/distilbert-service first"
        exit 1
    fi
    
    # Set permissions
    chown -R ${SERVICE_USER}:${SERVICE_USER} ${SERVICE_DIR}
    chmod +x ${SERVICE_DIR}/main.py
    
    print_success "Service files installed"
}

# Function to create Python virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    cd ${SERVICE_DIR}
    
    # Create virtual environment as service user
    sudo -u ${SERVICE_USER} python3 -m venv venv
    
    # Activate and install requirements
    sudo -u ${SERVICE_USER} bash -c "
        source venv/bin/activate && \
        pip install --upgrade pip && \
        pip install -r requirements.txt
    "
    
    print_success "Virtual environment created and dependencies installed"
}

# Function to create systemd service
create_systemd_service() {
    print_status "Creating systemd service..."
    
    cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=DistilBERT Campus Intent Classification Service
After=network.target
Wants=network.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${SERVICE_DIR}
Environment=PATH=${SERVICE_DIR}/venv/bin
ExecStart=${SERVICE_DIR}/venv/bin/python main.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

# Resource limits
LimitNOFILE=65536
LimitNPROC=32768

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${SERVICE_DIR} /var/log/${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable ${SERVICE_NAME}
    
    print_success "Systemd service created and enabled"
}

# Function to configure environment
setup_environment() {
    print_status "Setting up environment configuration..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "${SERVICE_DIR}/.env" ]; then
        cp ${SERVICE_DIR}/.env.example ${SERVICE_DIR}/.env
        
        # Update with VPS-specific settings
        sed -i "s/SERVICE_HOST=0.0.0.0/SERVICE_HOST=0.0.0.0/" ${SERVICE_DIR}/.env
        sed -i "s/SERVICE_PORT=8000/SERVICE_PORT=8000/" ${SERVICE_DIR}/.env
        sed -i "s/LOG_LEVEL=INFO/LOG_LEVEL=INFO/" ${SERVICE_DIR}/.env
        
        chown ${SERVICE_USER}:${SERVICE_USER} ${SERVICE_DIR}/.env
        chmod 600 ${SERVICE_DIR}/.env
        
        print_success "Environment configuration created"
    else
        print_warning "Environment file already exists"
    fi
}

# Function to setup firewall
setup_firewall() {
    print_status "Configuring firewall..."
    
    # Install UFW if not installed
    if ! command -v ufw &> /dev/null; then
        apt install -y ufw
    fi
    
    # Allow SSH (important!)
    ufw allow ssh
    
    # Allow DistilBERT service port
    ufw allow 8000/tcp comment 'DistilBERT Service'
    
    # Enable firewall
    ufw --force enable
    
    print_success "Firewall configured"
}

# Function to setup nginx reverse proxy (optional)
setup_nginx() {
    read -p "Do you want to setup Nginx reverse proxy? (y/N): " setup_nginx_choice
    
    if [[ $setup_nginx_choice =~ ^[Yy]$ ]]; then
        print_status "Setting up Nginx reverse proxy..."
        
        # Install Nginx
        apt install -y nginx
        
        # Get domain name
        read -p "Enter your domain name (e.g., distilbert.yourdomain.com): " domain_name
        
        if [ -z "$domain_name" ]; then
            domain_name="localhost"
        fi
        
        # Create Nginx configuration
        cat > /etc/nginx/sites-available/${SERVICE_NAME} << EOF
server {
    listen 80;
    server_name ${domain_name};
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
}
EOF
        
        # Enable site
        ln -sf /etc/nginx/sites-available/${SERVICE_NAME} /etc/nginx/sites-enabled/
        rm -f /etc/nginx/sites-enabled/default
        
        # Test and restart Nginx
        nginx -t && systemctl restart nginx && systemctl enable nginx
        
        print_success "Nginx reverse proxy configured for ${domain_name}"
    fi
}

# Function to start service
start_service() {
    print_status "Starting DistilBERT service..."
    
    # Start the service
    systemctl start ${SERVICE_NAME}
    
    # Wait a moment for startup
    sleep 5
    
    # Check service status
    if systemctl is-active --quiet ${SERVICE_NAME}; then
        print_success "DistilBERT service started successfully!"
        
        # Show service status
        systemctl status ${SERVICE_NAME} --no-pager -l
        
        # Test the service
        print_status "Testing service..."
        sleep 10
        
        if curl -f http://localhost:8000/health &>/dev/null; then
            print_success "Service is responding to health checks!"
        else
            print_warning "Service might still be starting up. Check logs with: journalctl -u ${SERVICE_NAME} -f"
        fi
    else
        print_error "Failed to start DistilBERT service"
        print_status "Check logs with: journalctl -u ${SERVICE_NAME} -f"
        exit 1
    fi
}

# Function to show final instructions
show_final_instructions() {
    print_success "🎉 DistilBERT Service setup completed!"
    echo
    echo "📋 Service Information:"
    echo "   - Service Name: ${SERVICE_NAME}"
    echo "   - Service User: ${SERVICE_USER}"
    echo "   - Service Directory: ${SERVICE_DIR}"
    echo "   - Service URL: http://localhost:8000"
    echo "   - Health Check: http://localhost:8000/health"
    echo "   - API Documentation: http://localhost:8000/docs"
    echo
    echo "🔧 Useful Commands:"
    echo "   - Start service: sudo systemctl start ${SERVICE_NAME}"
    echo "   - Stop service: sudo systemctl stop ${SERVICE_NAME}"
    echo "   - Restart service: sudo systemctl restart ${SERVICE_NAME}"
    echo "   - Check status: sudo systemctl status ${SERVICE_NAME}"
    echo "   - View logs: sudo journalctl -u ${SERVICE_NAME} -f"
    echo
    echo "📂 Important Files:"
    echo "   - Configuration: ${SERVICE_DIR}/.env"
    echo "   - Logs: /var/log/${SERVICE_NAME}/"
    echo "   - Service file: /etc/systemd/system/${SERVICE_NAME}.service"
    echo
    echo "🔍 Test the API:"
    echo "   curl -X POST http://localhost:8000/classify \\"
    echo "        -H 'Content-Type: application/json' \\"
    echo "        -d '{\"text\": \"Kapan jadwal kuliah besok?\"}'"
    echo
    if [[ $setup_nginx_choice =~ ^[Yy]$ ]]; then
        echo "🌐 Nginx Configuration:"
        echo "   - Domain: ${domain_name}"
        echo "   - Nginx Config: /etc/nginx/sites-available/${SERVICE_NAME}"
        echo "   - URL: http://${domain_name}"
    fi
}

# Main execution
main() {
    print_status "🚀 Starting DistilBERT Service VPS Setup..."
    
    # Check if running as root
    check_root
    
    # Execute setup steps
    update_system
    install_python
    create_service_user
    create_directories
    install_service_files
    create_venv
    setup_environment
    create_systemd_service
    setup_firewall
    setup_nginx
    start_service
    show_final_instructions
    
    print_success "✅ Setup completed successfully!"
}

# Run main function
main "$@"
