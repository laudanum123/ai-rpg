/**
 * Main JavaScript for AI Game Master RPG
 */

document.addEventListener('DOMContentLoaded', function() {
    // Flash message dismissal
    const flashMessages = document.querySelectorAll('.flash');
    flashMessages.forEach(function(message) {
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '&times;';
        closeBtn.className = 'flash-close';
        closeBtn.addEventListener('click', function() {
            message.remove();
        });
        message.appendChild(closeBtn);
    });
    
    // Character class selection in character creation
    const classSelect = document.getElementById('character_class');
    if (classSelect) {
        const classDescriptions = {
            fighter: document.getElementById('fighter-desc'),
            wizard: document.getElementById('wizard-desc'),
            rogue: document.getElementById('rogue-desc')
        };
        
        // Hide all descriptions initially
        Object.values(classDescriptions).forEach(desc => {
            if (desc) desc.style.display = 'none';
        });
        
        // Show description based on selection
        classSelect.addEventListener('change', function() {
            Object.values(classDescriptions).forEach(desc => {
                if (desc) desc.style.display = 'none';
            });
            
            const selectedClass = classSelect.value;
            if (selectedClass && classDescriptions[selectedClass]) {
                classDescriptions[selectedClass].style.display = 'block';
            }
        });
    }
    
    // Game log scrolling
    const gameLog = document.getElementById('game-log');
    if (gameLog) {
        // Scroll to bottom initially
        gameLog.scrollTop = gameLog.scrollHeight;
        
        // Add mutation observer to scroll to bottom when content changes
        const observer = new MutationObserver(function() {
            gameLog.scrollTop = gameLog.scrollHeight;
        });
        
        observer.observe(gameLog, { childList: true });
    }
    
    // Action input handling
    const actionText = document.getElementById('action-text');
    const submitAction = document.getElementById('submit-action');
    
    if (actionText && submitAction) {
        // Submit on Enter (but allow Shift+Enter for new line)
        actionText.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitAction.click();
            }
        });
    }
    
    // Quick action buttons
    const quickActionButtons = document.querySelectorAll('.action-btn');
    quickActionButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            if (actionText) {
                const action = this.getAttribute('data-action');
                switch(action) {
                    case 'look':
                        actionText.value = 'I look around the area carefully.';
                        break;
                    case 'inventory':
                        actionText.value = 'I check my inventory.';
                        break;
                    case 'rest':
                        actionText.value = 'I want to rest and recover.';
                        break;
                    case 'search':
                        actionText.value = 'I search the area for anything interesting.';
                        break;
                    case 'attack':
                        actionText.value = 'I attack the nearest enemy.';
                        break;
                    case 'defend':
                        actionText.value = 'I take a defensive stance.';
                        break;
                    case 'flee':
                        actionText.value = 'I try to flee from combat.';
                        break;
                    default:
                        actionText.value = action;
                }
                
                if (submitAction) {
                    submitAction.click();
                }
            }
        });
    });
}); 